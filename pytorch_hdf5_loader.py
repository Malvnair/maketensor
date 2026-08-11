"""
pytorch_hdf5_loader.py 

reads image sequences from multiple HDF5 shards and converts each sample into tensors. For every sample, it can generate a new synthetic TNO in memory using the stored observational metadata, then constructs three input channels per frame: science image, inverse-variance weight map, and valid-pixel mask.

"""


import os
import time
import argparse
import multiprocessing as mp
import numpy as np
from pathlib import Path

import torch
from torch.utils import data

from ptsemseg.loader.tno_injection import inject_cutout_sequence


CLASSY_BAD_BITS = (
    (1 << 0)   # BAD
    | (1 << 1) # SAT
    | (1 << 2) # INTRP
    | (1 << 3) # CR
    | (1 << 4) # EDGE
    | (1 << 7) # SUSPECT
    | (1 << 8) # NO_DATA
    | (1 << 9) # BRIGHT_OBJECT
    | (1 << 10) # CLIPPED
    | (1 << 14) # SENSOR_EDGE
    | (1 << 15) # UNMASKEDNAN
)


def derive_classy_valid_mask(science, variance, mask):
    """Return binary network-validity mask from raw uint16 CLASSY bits."""
    mask_bits = np.asarray(mask, dtype=np.uint16)
    sci = np.asarray(science)
    var = np.asarray(variance)
    bad = (mask_bits & CLASSY_BAD_BITS) != 0

    return (
        ~bad
        & np.isfinite(sci)
        & np.isfinite(var)
        & (var > 0)
    )


def _require_h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ShardDataset requires h5py. Install project dependencies with "
            "`python -m pip install -e .`."
        ) from exc
    return h5py


"""
Define a dataset that reads samples distributed across the shards
"""

class ShardDataset(data.Dataset):
    # locate shard and read the coutns
    def __init__(self, data_dir, base_seed=0, inject=True, num_implants=1, output_mode="normal"):
        h5py = _require_h5py()
        self.shard_paths = sorted(Path(data_dir).glob("*.h5"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No HDF5 shards found ")

        self.base_seed = base_seed
        # whether to plant a fake TNO into each loaded sample
        self.inject = inject
        self.num_implants = num_implants
        self.output_mode = output_mode

        # store current training epoch 
        self._epoch = mp.Value("i", 0)

        # Read number of samples
        counts = []
        for p in self.shard_paths:
            with h5py.File(p, "r") as f:
                counts.append(int(f.attrs["n_samples"]))
        self.counts = np.asarray(counts)
        # calculate which index for each sample
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)])
        self._handles = {}
        self._meta = {}
        self._owner_pid = os.getpid()

    # return length of samples across all shards
    def __len__(self):
        return int(self.offsets[-1])
    
    # remove open handles (FIX BUG)
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_meta"] = {}
        return state

    # close every file currently cached in the process
    def close(self):
        """Close any open handles. Call before creating DataLoader
        workers so no HDF5 file is open in the parent at fork time."""
        for f in self._handles.values():
            try:
                f.close()
            except Exception:
                pass
        self._handles.clear()
        self._meta.clear()

    def __del__(self):
        self.close()
        
    # each epoch generate different randomized tno paramters
    def set_epoch(self, epoch):
        """Call once per epoch so planted objects differ across epochs."""
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    # return cached handle for shard
    def _get_file(self, shard_idx):
        pid = os.getpid()
        if pid != self._owner_pid:
            self._handles = {}
            self._meta = {}
            self._owner_pid = pid
        # look for an already open hanfle
        f = self._handles.get(shard_idx)
        if f is None:
            h5py = _require_h5py()
            f = h5py.File(self.shard_paths[shard_idx], "r")
            self._handles[shard_idx] = f
        return f

    # static frame/template metadata, read once per worker per shard
    def _get_meta(self, shard_idx):
        meta = self._meta.get(shard_idx)
        if meta is None:
            f = self._get_file(shard_idx)
            meta = {
                "cent_time": f["frame/cent_time"][:],
                "zp": f["frame/zp"][:],
                "exptime": f["frame/exptime"][:],
                "gain": f["frame/gain"][:],
                "pixel_scale": f["frame/pixel_scale"][:],
                "psf_file": f["frame/psf_file"][:],
                "tmpl_cent_time": f["template/cent_time"][:],
                "tmpl_zp": f["template/zp"][:],
                "tmpl_exptime": f["template/exptime"][:],
                "tmpl_pixel_scale": f["template/pixel_scale"][:],
                "tmpl_psf_file": f["template/psf_file"][:],
            }
            self._meta[shard_idx] = meta
        return meta

    # Random number generator for 1 sample
    def _derive_rng(self, global_index):
        """
        Random TNO’s orbit, motion, brightness, and position.
        """
        ss = np.random.SeedSequence([self.base_seed, self._epoch.value,
                                     int(global_index)])
        # same sample and epoch always produce the same random values
        return np.random.Generator(np.random.Philox(ss))

    
    
    
    # find one training sample from the HDF5 dataset 
    def __getitem__(self, global_index):
        # which shard has the requested sample?
        shard_idx = int(np.searchsorted(self.offsets, global_index,
                                        side="right") - 1)
        # find index
        local_idx = int(global_index - self.offsets[shard_idx])

        f = self._get_file(shard_idx)

        # Load the images, variance maps, masks for the sample
        sci = np.asarray(f["science"][local_idx], dtype=np.float32)
        var = np.asarray(f["variance"][local_idx], dtype=np.float32)
        msk = np.asarray(f["mask"][local_idx])

        # metadata
        meta = dict(self._get_meta(shard_idx))
        meta["cutout_center"] = f["cutout_center"][local_idx]
        meta["ref_to_frame_affine"] = np.asarray(
            f["ref_to_frame_affine"][local_idx], dtype=np.float64)

        # Random synthetic variables, planted into the cutout
        if self.inject:
            rng = self._derive_rng(global_index)
            sci, var, msk, labels = inject_cutout_sequence(
                science=sci,
                variance=var,
                mask=msk,
                metadata=meta,
                rng=rng,
                num_implants=self.num_implants,
                output_mode=self.output_mode,
            )
        else:
            # raw background only, no fake object
            labels = {"num_implants": np.int64(0)}

        # HDF5 mask is the raw CLASSY uint16 bitfield. The model receives a
        # derived binary validity channel; DETECTED bits remain usable pixels.
        valid_bool = derive_classy_valid_mask(sci, var, msk)
        valid = valid_bool.astype(np.float32)
        weight = np.zeros_like(var, dtype=np.float32)
        weight[valid_bool] = 1.0 / var[valid_bool]

        med = np.median(weight[valid_bool]) if np.any(valid_bool) else 1.0
        if med > 0:
            weight /= med

        # Combine into network input and return as tensor
        x = np.stack([sci, weight, valid], axis=1)  # (T, 3, H, W)
        return {
            "input": torch.from_numpy(x),
            "labels": labels,
        }


####################
# Benchmark
##############


# Seb said to test to see hoW quickly they can produce batches
def bench_one(dataset, batch_size, num_workers, n_batches):
    dataset.close()
    
    
    # CREATE THE DATALOADER SESSION
    loader = data.DataLoader(
        dataset,
        # How many samples grouped
        batch_size=batch_size,
        # change order every spoch
        shuffle=True,
        # cpus
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    
    # batches loaded one at a time
    it = iter(loader)
    t0 = time.perf_counter()

    try:
        batch = next(it)
    except StopIteration:
        print("fail dataset small."
        )
        return

    # Record how long the first batch took to load
    t_first = time.perf_counter() - t0
    x = batch["input"]
    bytes_per_batch = x.element_size() * x.nelement()

    # reset counter
    n_done = 0
    t0 = time.perf_counter()

    for batch in it:
        n_done += 1
        if n_done >= n_batches:
            break
    # time for batches
    elapsed = time.perf_counter() - t0

    if n_done == 0:
        print(
            f" Not enough samples"
        )
        return

    
    # calcuslate how many megabytes of tensor data and samples loaded
    sps = n_done * batch_size / elapsed
    mbps = n_done * bytes_per_batch / elapsed / 1e6

    print(
        f"workers={num_workers:2d} first batch {t_first:6.2f}s   "
        f"steady: {sps:8.1f} samples/s, {mbps:8.1f} MB/s "
        f"({n_done} batches)"
    )


####################
# DEBUGGGGG
##############


def debug_injection(data_dir, seed, num_implants):
    """
    Runs injection sanity checks on sample 0, then hands the picture off
    to save_output_figure() in debug (fake-only) mode.
    """
    ds = ShardDataset(data_dir, base_seed=seed, inject=False)
    f = ds._get_file(0)

    # raw arrays kept aside so we can compare before/after
    sci_raw = np.asarray(f["science"][0], dtype=np.float32)
    var_raw = np.asarray(f["variance"][0], dtype=np.float32)
    msk_raw = np.asarray(f["mask"][0])

    meta = dict(ds._get_meta(0))
    meta["ref_to_frame_affine"] = np.asarray(
        f["ref_to_frame_affine"][0], dtype=np.float64)

    def fresh_rng(epoch):
        ss = np.random.SeedSequence([seed, epoch, 0])
        return np.random.Generator(np.random.Philox(ss))

    sci, var, msk, labels = inject_cutout_sequence(
        science=sci_raw.copy(), variance=var_raw.copy(), mask=msk_raw.copy(),
        metadata=meta, rng=fresh_rng(0), num_implants=num_implants,
        output_mode="debug",
    )

    print("\n injected positions (crop coords, frame by frame):")
    for j in range(num_implants):
        print(f"\n implant {j}:")
        for i, (x, y) in enumerate(labels["positions_crop"][j]):
            print(f"   frame {i}: ({x:.1f}, {y:.1f})")
        print(
            f" mag={labels['mag'][j]:.2f}  "
            f"rate={labels['rate'][j]:.2f} arcsec/hr"
        )
        
    ds.close()
    save_output_figure(data_dir, seed, num_implants, "debug")


####################
# SAVE FIGGGGG
##############


def save_output_figure(data_dir, seed, num_implants, output_mode):
    """
    Saves one sample using the selected output mode.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = ShardDataset(data_dir, base_seed=seed, inject=True,
                      num_implants=num_implants,
                      output_mode=output_mode)
    sample = ds[0]
    labels = sample["labels"]
    filename = f"sample_{output_mode}.png"

    if output_mode == "ground-truth":
        # single panel, the returned ground-truth target
        frames = [np.asarray(labels["track_target"], dtype=np.float32)]
        titles = ["ground-truth target"]
    else:
        # every science frame returned to the network (fake-only in debug mode)
        science = sample["input"][:, 0].numpy()
        n_science = science.shape[0]
        frames = [science[i] for i in range(n_science)]
        titles = [f"frame {i}" for i in range(n_science)]

        if output_mode in ("mean", "median"):
            # tno_injection already computed the stack at the right point,
            frames.append(np.asarray(labels["final_stack"], dtype=np.float32))
            titles.append(f"frame {n_science}   \u00b7   {output_mode} stack "
                          f"({n_science} frames)")

    n_panels = len(frames)
    seq = np.stack(frames, axis=0)

    if output_mode == "ground-truth":
        finite = np.isfinite(seq)
        vmax = np.nanmax(seq[finite]) if np.any(finite) else 1.0
        if vmax <= 0:
            vmax = 1.0
        vmin = 0.0
    else:
        # one display scale from the whole plotted sequence
        med = np.nanmedian(seq)
        mad = np.nanmedian(np.abs(seq - med))
        sigma = 1.4826 * mad if mad > 0 else np.nanstd(seq)
        vmin = med - 3 * sigma
        vmax = med + 5 * sigma

    fig, axes = plt.subplots(n_panels, 1, figsize=(4, 4 * n_panels))
    if n_panels == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.imshow(frames[i], origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(titles[i], fontsize=8, loc="left")
        ax.axis("off")

    header = (f"output_mode={output_mode} | num_implants={num_implants} | "
              f"seed={seed}")
    fig.suptitle(header, fontsize=10, y=0.999)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(filename, dpi=120)
    plt.close(fig)
    print(f" saved {filename}")

    ds.close()


####################
# Main
##############

    
    
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DataLoader throughput on raw HDF5 shards."
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory of *.h5 shards")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 8],
                        help="Worker counts to sweep")
    parser.add_argument("--n-batches", type=int, default=50,
                        help="Steady-state batches per config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-inject", action="store_true",
                        help="Skip TNO injection (raw loading speed only)")
    parser.add_argument("--num-implants", type=int, default=1,
                        help="Fake objects per sample when injecting")
    parser.add_argument("--output-mode", type=str, default="normal",
                        choices=("normal", "debug", "mean", "median",
                                 "ground-truth"),
                        help="Injection output to return")
    parser.add_argument("--save-fig", action="store_true",
                        help="Save one sample from the selected output mode "
                             "as a PNG and exit")
    parser.add_argument("--debug-injection", action="store_true",
                        help="Inject into sample 0, run sanity checks, save "
                             "a fake-only PNG, and exit")
    args = parser.parse_args()

    if args.debug_injection:
        debug_injection(args.data_dir, args.seed, args.num_implants)
        return

    if args.save_fig:
        if args.no_inject:
            parser.error("--save-fig requires injection; remove --no-inject")
        save_output_figure(args.data_dir, args.seed, args.num_implants,
                           args.output_mode)
        return

    ds = ShardDataset(args.data_dir, base_seed=args.seed,
                      inject=not args.no_inject,
                      num_implants=args.num_implants,
                      output_mode=args.output_mode)

    sample = ds[0]["input"]
    T, C, H, W = sample.shape
    per_sample_mb = sample.element_size() * sample.nelement() / 1e6
    ds.close()

    print("=" * 60)
    print("Shard loader benchmark")
    print("=" * 60)
    print(f"  Data dir:    {args.data_dir}")
    print(f"  Shards:      {len(ds.shard_paths)}")
    print(f"  Samples:     {len(ds)}")
    print(f"  Sample:      (T={T}, C={C}, {H}x{W})  ~{per_sample_mb:.2f} MB as tensor")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Injection:   {'OFF (raw loading only)' if args.no_inject else 'ON'}")
    print("=" * 60)

    for nw in args.workers:
        bench_one(ds, args.batch_size, nw, args.n_batches)


if __name__ == "__main__":
    main()
