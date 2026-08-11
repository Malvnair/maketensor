"""
fits2hdf5.py

Reads the original FITS files from shared ARC storage once and converts the real image sequences and metadata into organized HDF5 shards. This avoids repeatedly opening thousands of FITS files during training.


creates raw, unimplanted HDF5 cutouts from the original FITS files
"""

import sys
import argparse
import warnings
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
import h5py

sys.path.append("/arc/home/malvnair/trippy")
from trippy import psf as trippy_psf


####################
#Paths
##############

VISIT_LIST_ROOT = Path("/arc/projects/classy/visitLists")
WARP_ROOT       = Path("/arc/projects/classy/warps")
DB_ROOT         = Path("/arc/projects/classy/dbimages")

ARCSEC_PER_DEG  = 3600.0
PSF_STAMP_SIZE  = 64

FILTER_CODES = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4}
FILTER_UNKNOWN = 15

# type HDF5
STR_DT = h5py.string_dtype(encoding="utf-8")


####################
# Helpers (from maketensor8.py)
##############


def find_psf_file_from_dbimages(image_id, ccd):
    """Look up the TRIPPy PSF file for a given science image ID and CCD number."""
    image_id = str(image_id)
    path_ccd = DB_ROOT / image_id / f"ccd{ccd}" / f"{image_id}p{ccd}.psf.fits"
    path_top = DB_ROOT / image_id / f"{image_id}p{ccd}.psf.fits"
    if path_ccd.exists():
        return path_ccd
    if path_top.exists():
        return path_top
    return None


def encode_filter(header):
    # rename the filters so no crash
    raw = str(header.get("FILTER", "") or header.get("FILTER1", "") or "")
    band = raw.strip().lower()[:1]
    return FILTER_CODES.get(band, FILTER_UNKNOWN), raw


def read_seeing_fwhm(header):
    for key in ("SEEING", "IQFWHM", "FWHM", "SEEFWHM"):
        val = header.get(key)
        if val:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return 0.0


def wcs_header_string(wcs_obj):
    return wcs_obj.to_header(relax=True).tostring()


def build_wcs(header):
    """Build a WCS from a header, RADECSYS handling."""
    header = header.copy()
    if "RADECSYS" in header and "RADESYSa" not in header:
        header["RADESYSa"] = header["RADECSYS"]
    bad_pv = [k for k in header.keys()
              if k.startswith("PV") and "_" in k and
              int(k.split("_")[1]) >= 5]
    for k in bad_pv:
        del header[k]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WCS(header, naxis=2)


def pixel_scale(wcs_obj):
    """Return mean pixel scale in arcsec/pixel."""
    try:
        scales = wcs_obj.proj_plane_pixel_scales()
        return float(np.mean([s.value for s in scales]) * ARCSEC_PER_DEG)
    except Exception:
        return 0.185  # MegaCam pixel scale


def build_ref_to_frame_transform(wcs0, wcs_i, x0c, y0c):
    """
    Build a local matrix that maps referencecrop cordinates
    learns how one little piece of frame 0 has shifted, rotated in another exposure
    """
    px0 = np.array([x0c, x0c + 20.0, x0c])
    py0 = np.array([y0c, y0c, y0c + 20.0])
    ra, dec = wcs0.all_pix2world(px0, py0, 0)
    px, py = wcs_i.all_world2pix(ra, dec, 0)
    ax = (px[1] - px[0]) / 20.0
    bx = (px[2] - px[0]) / 20.0
    ay = (py[1] - py[0]) / 20.0
    by = (py[2] - py[0]) / 20.0
    return np.array([[ax, bx, px[0] - x0c],
                     [ay, by, py[0] - y0c]], dtype=np.float32)


def find_image_hdu(hdul):
    """Return data array as float for first 2D image HDU."""
    for hdu in hdul:
        # loop through until 2d image
        if hdu.data is not None and hdu.data.ndim == 2:
            return hdu.data.astype(float), hdu
    sys.exit("Not found")


def find_named_hdu(hdul, extname):
    for hdu in hdul:
        if hdu.data is None or getattr(hdu.data, "ndim", 0) != 2:
            continue

        extname_value = str(hdu.header.get("EXTNAME", "")).strip().upper()
        exttype_value = str(hdu.header.get("EXTTYPE", "")).strip().upper()

        if extname_value == extname or exttype_value == extname:
            return hdu.data

    return None


def get_zeropoint(hdul):
    """
    zp = -2.5 * log10(calibrationMean) + 31.4 
    """
    for hdu in hdul:
        if not hasattr(hdu, "columns"):
            continue
        col_names = [c.name for c in hdu.columns]
        if "calibrationMean" in col_names:
            cal_mean = float(hdu.data["calibrationMean"][0])
            if cal_mean > 0:
                return float(-2.5 * np.log10(cal_mean) + 31.4)
    for hdu in hdul:
        for zpv in ("PHOTZP", "MAGZP", "ZEROPT", "FLXMAG0"):
            val = hdu.header.get(zpv)
            if val:
                return float(val)
    print("No zeropoint found.")
    return 0.0


# Convert the full TRIPPy PSF into a normalized 64×64 pixel array that can be stored directly in HDF5
def make_psf_stamp(psf_file, stamp_size=PSF_STAMP_SIZE):
    """
    PSF stamp
    """
    psf_file = Path(psf_file)
    if not psf_file.exists():
        sys.exit(f"PSF file not found: {psf_file}")

    mpsf = trippy_psf.modelPSF(restore=str(psf_file))

    # Choose a large temporary canvas so  planted PSF is far from edges
    canvas_size = max(4 * stamp_size, 256)
    c = (canvas_size - 1) / 2.0
    canvas = np.zeros((canvas_size, canvas_size), dtype=float)
    # Plant one noiseless, unit-flux, untrailed PSF in the centre of the blank image.
    stamp_full = mpsf.plant(
        np.array([c]), np.array([c]), np.array([1.0]), canvas,
        useLinePSF=False, returnModel=True,
        gain=1.0, addNoise=False, verbose=False,
    )

    half = stamp_size // 2
    ci = int(round(c))
    stamp = stamp_full[ci - half:ci + half, ci - half:ci + half].copy()
    
    # Normalize the stamp so it describes only the PSF shape and has total flux equal to one.
    total = np.sum(stamp)
    if total > 0:
        stamp = stamp / total
    else:
        print("PSF stamp returned zeroflux")
    return stamp.astype(np.float32)


####################
# Visit / template loading (maketensor8's WarpDataset, flattened)
##############


def load_visit(visit, ccd, n_frames, sort_by_time=True):
    """
    Load the science DIFFEXP sequence for one visit/CCD into a dict of
    parallel lists.
    """
    visit_list_path = VISIT_LIST_ROOT / visit / f"{visit}_visit_list.txt"
    if not visit_list_path.exists():
        sys.exit(f"Visit list not found: {visit_list_path}")

    warp_dir = WARP_ROOT / visit / str(ccd)
    if not warp_dir.is_dir():
        sys.exit(f"Warp directory not found")

    template_ids = []
    with open(visit_list_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                template_ids.append(line)
    if not template_ids:
        sys.exit("Visit list is empty.")

    fits_files, image_ids = [], []
    for sci_id in template_ids:
        matches = sorted(warp_dir.glob(f"DIFFEXP-{sci_id}-*-{ccd}.fits"))
        if len(matches) == 1:
            fits_files.append(matches[0])
            image_ids.append(sci_id)
        elif len(matches) == 0:
            print(f"No DIFFEXP found.")
        else:
            print(f"Multiple DIFFEXP matches.")

    if not fits_files:
        sys.exit("No DIFFEXP files found in warp directory.")

    fits_files = fits_files[:n_frames]
    image_ids = image_ids[:n_frames]

    d = {k: [] for k in (
        "images", "variances", "masks", "mjds", "cent_times", "exptimes",
        "zeropoints", "gains", "wcs_list", "pixel_scales", "image_ids",
        "psf_files", "airmasses", "seeing_fwhms", "filter_codes",
        "filter_names", "wcs_headers",
    )}

    for fits_file, image_id in zip(fits_files, image_ids):
        with fits.open(fits_file) as hdul:
            image, img_hdu = find_image_hdu(hdul)
            variance = find_named_hdu(hdul, "VARIANCE")
            mask_raw = find_named_hdu(hdul, "MASK")

            img_header = img_hdu.header
            wcs_obj = build_wcs(img_header)
            zp = get_zeropoint(hdul)
            gain = float(img_header.get("GAIN") or 3.0)

            primary = hdul[0].header
            exptime = float(primary.get("EXPTIME") or
                            img_header.get("EXPTIME") or 0.0)
            if "MJD-OBS" in primary:
                mjd = float(primary["MJD-OBS"])
            elif "MJD-OBS" in img_header:
                mjd = float(img_header["MJD-OBS"])
            else:
                print(f"no MJD-OBS found in {fits_file.name}")
                mjd = 0.0

            cent_time = mjd + exptime / (2.0 * 86400.0)
            pscale = pixel_scale(wcs_obj)

            airmass = float(primary.get("AIRMASS") or
                            img_header.get("AIRMASS") or 0.0)
            seeing = read_seeing_fwhm(primary) or read_seeing_fwhm(img_header)
            filter_code, filter_name = encode_filter(primary)
            if filter_code == FILTER_UNKNOWN:
                filter_code, filter_name = encode_filter(img_header)
            wcs_hdr_str = wcs_header_string(wcs_obj)

        # If no variance map exists, store NaN 
        if variance is None:
            variance = np.full_like(image, np.nan, dtype=float)
            print(f"No VARIANCE HDU.")   
            
        if mask_raw is None:
            mask = np.zeros_like(image, dtype=np.uint16)
        else:
            mask = mask_raw.astype(np.uint16)
            
        psf_path = find_psf_file_from_dbimages(image_id, ccd)
        if psf_path is None:
            sys.exit(f"No PSF found for image_id {image_id}")

        d["images"].append(image)
        d["variances"].append(variance.astype(float))
        d["masks"].append(mask)
        d["mjds"].append(mjd)
        d["cent_times"].append(cent_time)
        d["exptimes"].append(exptime)
        d["zeropoints"].append(zp)
        d["gains"].append(gain)
        d["wcs_list"].append(wcs_obj)
        d["pixel_scales"].append(pscale)
        d["image_ids"].append(image_id)
        d["psf_files"].append(psf_path)
        d["airmasses"].append(airmass)
        d["seeing_fwhms"].append(seeing)
        d["filter_codes"].append(filter_code)
        d["filter_names"].append(filter_name)
        d["wcs_headers"].append(wcs_hdr_str)

    if sort_by_time:
        order = sorted(range(len(d["cent_times"])),
                       key=lambda k: d["cent_times"][k])
        for key in d:
            d[key] = [d[key][k] for k in order]

    for i, fid in enumerate(d["image_ids"]):
        print(f"  [{i}] id={fid}  zp={d['zeropoints'][i]:.4f}  "
              f"exptime={d['exptimes'][i]:.1f}s  mjd={d['mjds'][i]:.6f}  "
              f"psf={d['psf_files'][i].name}")
    return d


def load_templates(visit, ccd):
    """
    Metadata for the subtraction-template images (negative wells).
    """
    path = VISIT_LIST_ROOT / visit / f"{visit}_template_visit_list.txt"
    if not path.exists():
        sys.exit(f"Template visit list not found: {path}")

    ids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    if not ids:
        sys.exit("Template visit list is empty.")

    warp_dir = WARP_ROOT / visit / str(ccd)

    t = {k: [] for k in ("image_ids", "psf_files", "mjds", "cent_times",
                         "exptimes", "zeropoints", "pixel_scales")}

    for tmpl_id in ids:
        matches = sorted(warp_dir.glob(f"DIFFEXP-{tmpl_id}-*-{ccd}.fits"))
        if len(matches) == 0:
            print(f"skipping.")
            continue
        if len(matches) > 1:
            print(f"too many")
        fits_file = matches[0]

        psf_path = find_psf_file_from_dbimages(tmpl_id, ccd)
        if psf_path is None:
            print(f"No PSF")
            continue

        with fits.open(fits_file) as hdul:
            _image, img_hdu = find_image_hdu(hdul)
            img_header = img_hdu.header
            wcs_obj = build_wcs(img_header)
            zp = get_zeropoint(hdul)

            primary = hdul[0].header
            exptime = float(primary.get("EXPTIME") or
                            img_header.get("EXPTIME") or 0.0)
            if "MJD-OBS" in primary:
                mjd = float(primary["MJD-OBS"])
            elif "MJD-OBS" in img_header:
                mjd = float(img_header["MJD-OBS"])
            else:
                print(f"no MJD-OBS found in {fits_file.name}")
                mjd = 0.0

        t["image_ids"].append(tmpl_id)
        t["psf_files"].append(psf_path)
        t["mjds"].append(mjd)
        t["cent_times"].append(mjd + exptime / (2.0 * 86400.0))
        t["exptimes"].append(exptime)
        t["zeropoints"].append(zp)
        t["pixel_scales"].append(pixel_scale(wcs_obj))

    if not t["image_ids"]:
        sys.exit("No loaded.")
    return t


####################
# Cutout sampling
##############

MAX_TRIES = 50


def sample_center(rng, d, half_size):

    img0 = d["images"][0]
    wcs0 = d["wcs_list"][0]
    h, w = img0.shape
    margin = half_size + 2  # +2 so the int rounding in cutout() can't clip DEBUG

    for _ in range(MAX_TRIES):
        cx = rng.uniform(margin, w - margin)
        cy = rng.uniform(margin, h - margin)

        ra_arr, dec_arr = wcs0.all_pix2world([cx], [cy], 0)
        ra, dec = float(ra_arr[0]), float(dec_arr[0])

        ok = True
        for wcs_i, img in zip(d["wcs_list"], d["images"]):
            hi, wi = img.shape
            px_arr, py_arr = wcs_i.all_world2pix([ra], [dec], 0)
            px, py = float(px_arr[0]), float(py_arr[0])
            if not (margin <= px < wi - margin and margin <= py < hi - margin):
                ok = False
                break
        if ok:
            return float(cx), float(cy)

    raise RuntimeError("Could not find a safe cutout centre")


def cutout(image, x, y, half_size):
    """Same rounding as maketensor8"""
    h, w = image.shape
    cx = int(round(x))
    cy = int(round(y))
    x0, x1 = cx - half_size, cx + half_size
    y0, y1 = cy - half_size, cy + half_size
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    return image[y0:y1, x0:x1].copy()


####################
# Shard writing
##############


def write_shard(out_path, d, tmpl, centers, half_size, run_attrs,
                compression=None):
    """
    Write one flat shard
    """
    
    # M = number of samples or cutout sequences in the shard
    # T = number of time frames in each sequence
    # H, W = height, width of each cutout in pixels
    T = len(d["images"])
    M = len(centers)
    H = W = 2 * half_size
    wcs0 = d["wcs_list"][0]

    # key value datA
    with h5py.File(out_path, "w") as f:
        for k, v in run_attrs.items():
            f.attrs[k] = v

        chunk4 = (1, T, H, W)
        ds_sci = f.create_dataset("science", (M, T, H, W), dtype=np.float32,
                                  chunks=chunk4, compression=compression)
        ds_var = f.create_dataset("variance", (M, T, H, W), dtype=np.float32,
                                  chunks=chunk4, compression=compression)
        ds_msk = f.create_dataset("mask", (M, T, H, W), dtype=np.uint16,
                                  chunks=chunk4, compression=compression)
        ds_ctr = f.create_dataset("cutout_center", (M, 2), dtype=np.float32)
        ds_org = f.create_dataset("ref_pixel_origin", (M, 2), dtype=np.float32)
        ds_aff = f.create_dataset("ref_to_frame_affine", (M, T, 2, 3),
                                  dtype=np.float32)

        # For each sampled sky location, extract the same cutout sequence across all frames 
        for m, (x_ref, y_ref) in enumerate(centers):
            sci = np.stack([cutout(img, x_ref, y_ref, half_size)
                            for img in d["images"]])
            var = np.stack([cutout(v, x_ref, y_ref, half_size)
                            for v in d["variances"]])
            msk = np.stack([cutout(mk, x_ref, y_ref, half_size)
                            for mk in d["masks"]]).astype(np.uint16)




            cx = int(round(x_ref))
            cy = int(round(y_ref))

            ds_sci[m] = sci.astype(np.float32)
            ds_var[m] = var.astype(np.float32)
            ds_msk[m] = msk
            ds_ctr[m] = (x_ref, y_ref)
            ds_org[m] = (cx - half_size, cy - half_size)
            # Store how reference-frame coordinates map into every other frame            
            ds_aff[m] = np.stack([
                build_ref_to_frame_transform(wcs0, wcs_i, cx - half_size, cy - half_size)
                for wcs_i in d["wcs_list"]
            ])


        # per-frame metadata
        gf = f.create_group("frame")
        gf.create_dataset("mjd", data=np.asarray(d["mjds"], dtype=np.float64))
        gf.create_dataset("cent_time", data=np.asarray(d["cent_times"], dtype=np.float64))
        gf.create_dataset("exptime", data=np.asarray(d["exptimes"], dtype=np.float32))
        gf.create_dataset("zp", data=np.asarray(d["zeropoints"], dtype=np.float32))
        gf.create_dataset("gain", data=np.asarray(d["gains"], dtype=np.float32))
        gf.create_dataset("pixel_scale", data=np.asarray(d["pixel_scales"], dtype=np.float32))
        gf.create_dataset("airmass", data=np.asarray(d["airmasses"], dtype=np.float32))
        gf.create_dataset("seeing_fwhm", data=np.asarray(d["seeing_fwhms"], dtype=np.float32))
        gf.create_dataset("filter", data=np.asarray(d["filter_codes"], dtype=np.uint8))
        gf.create_dataset("image_id", data=[str(s) for s in d["image_ids"]], dtype=STR_DT)
        gf.create_dataset("filter_name", data=[str(s) for s in d["filter_names"]], dtype=STR_DT)
        gf.create_dataset("psf_file", data=[str(p) for p in d["psf_files"]], dtype=STR_DT)
        gf.create_dataset("wcs_header", data=list(d["wcs_headers"]), dtype=STR_DT)
        # native untrailed PSF stamp per frame (contract 7.1)
        gf.create_dataset(
            "psf_stamp",
            data=np.stack([make_psf_stamp(p) for p in d["psf_files"]]),
        )
        gf.attrs["zp_units"] = "counts_total" 

        # template metadata for negative wells
        gt = f.create_group("template")
        gt.create_dataset("image_id", data=[str(s) for s in tmpl["image_ids"]], dtype=STR_DT)
        gt.create_dataset("psf_file", data=[str(p) for p in tmpl["psf_files"]], dtype=STR_DT)
        gt.create_dataset("mjd", data=np.asarray(tmpl["mjds"], dtype=np.float64))
        gt.create_dataset("cent_time", data=np.asarray(tmpl["cent_times"], dtype=np.float64))
        gt.create_dataset("exptime", data=np.asarray(tmpl["exptimes"], dtype=np.float32))
        gt.create_dataset("zp", data=np.asarray(tmpl["zeropoints"], dtype=np.float32))
        gt.create_dataset("pixel_scale", data=np.asarray(tmpl["pixel_scales"], dtype=np.float32))
        gt.create_dataset(
            "psf_stamp",
            data=np.stack([make_psf_stamp(p) for p in tmpl["psf_files"]]),
        )


####################
# Main
##############


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert DIFFEXP FITS sequences into raw unplanted HDF5 shards."
    )
    parser.add_argument("--visit", type=str, required=True,
                        help="Visit name, e.g. 2022-08-01-AS1_July")
    parser.add_argument("--ccd", type=int, default=15, help="CCD number")
    parser.add_argument("--num-samples", type=int, default=256,
                        help="Total raw cutout samples to generate")
    parser.add_argument("--samples-per-shard", type=int, default=64,
                        help="Samples per shard file")
    parser.add_argument("--n-frames", type=int, default=8,
                        help="Max science frames to load")
    parser.add_argument("--half-size", type=int, default=50,
                        help="Cutout half-size in pixels")
    parser.add_argument("--out-dir", type=str, default="/arc/home/malvnair/shards_raw",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed for cutout-centre sampling")
    parser.add_argument("--gzip", action="store_true",
                        help="gzip-compress the big datasets (slower reads)")
    parser.add_argument("--keep-template-order", action="store_true",
                        help="Use visit-list order instead of MJD order")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Raw (unplanted) HDF5 shard generator")
    print("=" * 60)
    print(f"  Visit:             {args.visit}")
    print(f"  CCD:               {args.ccd}")
    print(f"  Num samples:       {args.num_samples}")
    print(f"  Samples per shard: {args.samples_per_shard}")
    print(f"  Frames:            {args.n_frames}")
    print(f"  Half-size:         {args.half_size} px")
    print(f"  Output dir:        {out_dir}")
    print(f"  Compression:       {'gzip' if args.gzip else 'none'}")
    print("=" * 60)

    
    d = load_visit(args.visit, args.ccd, args.n_frames,
                   sort_by_time=not args.keep_template_order)
    tmpl = load_templates(args.visit, args.ccd)

    compression = "gzip" if args.gzip else None
    n_shards = int(np.ceil(args.num_samples / args.samples_per_shard))
    n_written = 0
    
    # how many shards?
    for k in range(n_shards):
        m_this = min(args.samples_per_shard, args.num_samples - n_written)

        rng = np.random.default_rng(
            np.random.SeedSequence([args.seed, args.ccd, k])
        )

        centers = []
        for _ in range(m_this):
            centers.append(sample_center(rng, d, args.half_size))

        run_attrs = {
            "visit": args.visit,
            "ccd": int(args.ccd),
            "shard_idx": int(k),
            "n_samples": int(m_this),
            "n_frames": int(len(d["images"])),
            "half_size": int(args.half_size),
            "seed": int(args.seed),
            "planted": 0,   # raw shards: nothing injected
            "image_kind": "diffexp",  # deviation from contract 7.2, ASK WES/SEB
        }

        out_path = out_dir / (f"{args.visit}_ccd{args.ccd}_"
                              f"shard{k:04d}.h5")
        write_shard(out_path, d, tmpl, centers, args.half_size, run_attrs,
                    compression=compression)
        n_written += m_this

    print(f"\n Done: {n_written} samples across {n_shards} shards in {out_dir}")


if __name__ == "__main__":
    main()
