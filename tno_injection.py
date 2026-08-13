"""
tno_injection.py

Reused synthetic-TNO logic from maketensor8.py


"""

import numpy as np
from pathlib import Path

# trippy import is lazy so the module can be imported without it installed
trippy_psf = None


def _import_trippy():
    global trippy_psf
    if trippy_psf is None:
        from trippy import psf as _tp
        trippy_psf = _tp


# PhotoCalib
# Handles zeropoint reading and magnitude→count conversion.

class PhotoCalib:
    """
    Magnitude to photometric zeropoint to image counts (ADU)
    """

    # converting from nJY
    PHOTOCALIB_ZP_OFFSET = 31.4

    @staticmethod
    def get_zeropoint(hdul):
        """
        Read the zeropoint from a FITS HDU list.
        """
        # primary method: PhotoCalib BinTable
        for hdu in hdul:
            if not hasattr(hdu, "columns"):
                continue
            # make list of all column names and checks if name in it
            col_names = [c.name for c in hdu.columns]
            if "calibrationMean" in col_names:
                cal_mean = float(hdu.data["calibrationMean"][0])
                if cal_mean > 0:
                    zp = -2.5 * np.log10(cal_mean) + PhotoCalib.PHOTOCALIB_ZP_OFFSET
                    return float(zp)

        # simple header keywords
        for hdu in hdul:
            for zpv in ("PHOTZP", "MAGZP", "ZEROPT", "FLXMAG0"):
                val = hdu.header.get(zpv)
                if val:
                    return float(val)

        print("No zeropoint found.")
        return 0.0

    @staticmethod
    def mag_to_counts(mag, zeropoint):
        """
        Convert AB magnitude to expected total count (ADU).
        """
        return 10.0 ** ((zeropoint - mag) / 2.5)


########
# OrbitSampler
# Generates random synthetic TNO orbital parameters.

class OrbitSampler:
    """
    Randomly makes heliocentric distance, eccentricity, and inclination
    for a synthetic KBO.

    alpha_rad = solar elongation angle
    theta_rad = true anomaly. near-opposition default
    """

    # Classical KBO values
    R_MIN = 30.0   # Neptune's orbit / inner classical belt (AU)
    R_MAX = 50.0   # outer classical belt edge  ASK WES/SEB for survey range
    E_MAX = 0.25
    I_MAX_DEG = 35.0  # degrees

    def __init__(self, r_min=None, r_max=None, e_max=None, i_max_deg=None, rng=None):
        self.r_min = r_min or self.R_MIN
        self.r_max = r_max or self.R_MAX
        self.e_max = e_max or self.E_MAX
        self.i_max_deg = i_max_deg or self.I_MAX_DEG
        self.rng = rng or np.random.default_rng()

    def sample(self):
        """Return one randomly drawn set of orbital parameters as a dict."""
        # randomly generate the calues
        r = self.rng.uniform(self.r_min, self.r_max)
        e = self.rng.uniform(0.0, self.e_max)
        i_deg = self.rng.uniform(0.0, self.i_max_deg)
        i_rad = np.deg2rad(i_deg)

        # Near-opposition approximation Appendix B
        alpha_rad = 0.0
        theta_rad = 0.0

        return {
            "r": r,
            "e": e,
            "i_deg": i_deg,
            "i_rad": i_rad,
            "alpha_rad": alpha_rad,
            "theta_rad": theta_rad,
        }

################
# Converts orbital parameters to sky-plane motion rates.
################

class MotionModel:
    """
    Computes apparent sky-plane motion of a TNO using the Appendix B
    """
    EARTH_ANGULAR_VELOCITY = 148.0   # arcsec/hr

    @staticmethod
    def compute(orbit):
        """
        Returns dict with rate_ra, rate_dec, rate, angle.
        """
        r = orbit["r"]
        i = orbit["i_rad"]
        alpha = orbit["alpha_rad"]
        theta = orbit["theta_rad"]

        # geocentric distance approximation
        delta = r - 1.0
        if delta <= 0.0:
            delta = 0.1   # GET RIF OF DIVISION BY ZERO ERROR

        # Appendix B equations
        raw_x = np.cos(alpha) / delta - np.cos(i) * np.cos(alpha + theta) / (r ** 1.5)
        raw_y = np.sin(i) / (r ** 1.5)

        # convert to arcsec/hr
        rate_ra  =  MotionModel.EARTH_ANGULAR_VELOCITY * raw_x
        rate_dec =  MotionModel.EARTH_ANGULAR_VELOCITY * raw_y

        # combined speed
        rate  = np.sqrt(rate_ra**2 + rate_dec**2)
        # Position anfle measured from N through E
        angle = np.degrees(np.arctan2(rate_ra, rate_dec)) % 360.0


        return {
            "rate_ra": rate_ra,
            "rate_dec": rate_dec,
            "rate": rate,
            "angle": angle,
        }

##############
# Draw a random apparent magnitude for the synthetic TNO
########

class MagnitudeSampler:
    """
    Draws a random AB magnitude for the fake TNO.
    """

    # Should be 27 but for testing is very bright
    MAG_MIN = 22.0
    MAG_MAX = 24.0

    def __init__(self, mag_min=None, mag_max=None, rng=None):
        self.mag_min = mag_min or self.MAG_MIN
        self.mag_max = mag_max or self.MAG_MAX
        self.rng = rng or np.random.default_rng()

    def sample(self):
        # Randomm
        return float(self.rng.uniform(self.mag_min, self.mag_max))


##############
# TRIPPy rendering 
# restore the PSF, build the trailed line PSF, plant at (x, y)
########

def _as_path(p):
    # HDF5 hands strings back as bytes, make sure we get a plain str path
    if isinstance(p, bytes):
        p = p.decode("utf-8")
    return str(p)


def build_line_psf(psf_file, rate, rate_ra, rate_dec, exptime, pixel_scale):
    """
    Restore a TRIPPy PSF from file and apply line() for this exposure.
    """
    _import_trippy()
    psf_file = Path(_as_path(psf_file))
    if not psf_file.exists():
        raise FileNotFoundError(f"PSF file not found: {psf_file}")

    # restore PSF fresh each call
    mpsf = trippy_psf.modelPSF(restore=str(psf_file))

    r2d = 180.0 / np.pi
    # Convert ra/dec motion to pixel angle
    trippy_angle = np.arctan2(-rate_dec, rate_ra) * r2d + 180.0

    if trippy_angle > 90:
        trippy_angle -= 180
    if trippy_angle < -90:
        trippy_angle += 180.0

    mpsf.line(
        rate,                   # total rate in arcsec/hr
        trippy_angle,           # TRIPPy pixel-space angle
        exptime / 3600.0,       # exposure duration in hours
        pixScale=pixel_scale,
        useLookupTable=True,
    )
    return mpsf


def make_psf_image(mpsf, shape, x, y, counts):
    """
    Plant the trailed PSF at (x, y) at unit amplitude, measure the flux,
    then scale so the total flux equals counts. Returns the noiseless model.
    """
    H, W = shape
    pad = 100

    big_shape = (H + 2 * pad, W + 2 * pad)

    # plant at unit amplitude on the padded blank canvas
    p_im = mpsf.plant(
        np.array([x + pad]),
        np.array([y + pad]),
        np.array([1.0]),
        np.zeros(big_shape, dtype=float),
        useLinePSF=True,
        returnModel=True,
        gain=1.0,
        addNoise=False,
        verbose=False,
    )


    norm_flux = np.sum(p_im)
    if norm_flux <= 0:
        print("PSF plant returned zeroflux")
        return np.zeros(shape, dtype=float)

    amp = counts / norm_flux
    # Scale fake source
    p_im *= amp

    # crop back to the training cutout
    return p_im[pad:pad + H, pad:pad + W]


##############
# Visible-position sampling
# reject frame-0 positions whose propagated track leaves any science frame
########

MAX_POSITION_TRIES = 100
MAX_ORBIT_TRIES = 20


def sample_visible_reference_position(rng, motion, affines, cent_times,
                                      cent_time0, pscales, H, W,
                                      pad_x, pad_y):
    """
    Pick a crop position so the TNO track stays inside
    the cutout in every science frame. 
    """
    T = len(cent_times)

    for _ in range(MAX_POSITION_TRIES):
        # candidate frame-0 position, away from the edges
        u_ref = rng.uniform(pad_x, W - pad_x)
        v_ref = rng.uniform(pad_y, H - pad_y)

        ok = True
        for i in range(T):
            # same affine + motion propagation as the injection loop
            a = affines[i]
            x_i = a[0, 0] * u_ref + a[0, 1] * v_ref + a[0, 2]
            y_i = a[1, 0] * u_ref + a[1, 1] * v_ref + a[1, 2]

            dt_hr = (cent_times[i] - cent_time0) * 24.0
            x_inj = x_i + motion["rate_ra"] * dt_hr / pscales[i]
            y_inj = y_i - motion["rate_dec"] * dt_hr / pscales[i]

            # must sit safely inside every frame
            if not (pad_x <= x_inj < W - pad_x and pad_y <= y_inj < H - pad_y):
                ok = False
                break

        if ok:
            return float(u_ref), float(v_ref)

    return None


##############
# inject_cutout_sequence
# The one public entry point: plant fake TNOs into a cutout sequence.
########

# what output_mode is allowed to be
VALID_OUTPUT_MODES = {"normal", "mean", "median", "debug", "ground-truth"}


def inject_cutout_sequence(science, variance, mask, metadata, rng,
                           num_implants=1,
                           add_negative_wells=True,
                           add_poisson_noise=True,
                           mag_min=None, mag_max=None,
                           edge_pad_frac=0.1,
                           output_mode="normal"):

    if output_mode not in VALID_OUTPUT_MODES:
        raise ValueError(
            f"Invalid output_mode {output_mode!r}; "
            f"expected one of {sorted(VALID_OUTPUT_MODES)}"
        )

    T, H, W = science.shape

    cent_times = np.asarray(metadata["cent_time"], dtype=float)
    zps        = np.asarray(metadata["zp"], dtype=float)
    exptimes   = np.asarray(metadata["exptime"], dtype=float)
    gains      = np.asarray(metadata["gain"], dtype=float)
    pscales    = np.asarray(metadata["pixel_scale"], dtype=float)
    psf_files  = [_as_path(p) for p in metadata["psf_file"]]
    affines    = np.asarray(metadata["ref_to_frame_affine"], dtype=float)

    # one fixed scale for the reference grid
    ref_pscale = float(pscales[0])

    if add_negative_wells:
        tmpl_ct        = np.asarray(metadata["tmpl_cent_time"], dtype=float)
        tmpl_zp        = np.asarray(metadata["tmpl_zp"], dtype=float)
        tmpl_expt      = np.asarray(metadata["tmpl_exptime"], dtype=float)
        tmpl_pscale    = np.asarray(metadata["tmpl_pixel_scale"], dtype=float)
        tmpl_psf_files = [_as_path(p) for p in metadata["tmpl_psf_file"]]
        n_tmpl = len(tmpl_ct)
    else:
        n_tmpl = 0

    # debug: dump the real background, keep only the fake signal
    if output_mode == "debug":
        science = np.zeros_like(science)

    # ground-truth: clean positive only track
    if output_mode == "ground-truth":
        track_target = np.zeros((H, W), dtype=np.float32)
    else:
        track_target = None

    # frame-0 midpoint as the timing reference, same as maketensor8
    cent_time0 = cent_times[0]

    # samplers share the one rng so everything is reproducible
    orbit_sampler = OrbitSampler(rng=rng)
    mag_sampler = MagnitudeSampler(mag_min=mag_min, mag_max=mag_max, rng=rng)

    # how close to the crop edge the frame-0 position is allowed to sit
    pad_x = edge_pad_frac * (W / 2.0)
    pad_y = edge_pad_frac * (H / 2.0)

    # per-implant label arrays, fixed shapes so PyTorch can collate them
    imp_r        = np.zeros(num_implants, dtype=np.float32)
    imp_e        = np.zeros(num_implants, dtype=np.float32)
    imp_i_deg    = np.zeros(num_implants, dtype=np.float32)
    imp_mag      = np.zeros(num_implants, dtype=np.float32)
    imp_rate     = np.zeros(num_implants, dtype=np.float32)
    imp_angle    = np.zeros(num_implants, dtype=np.float32)
    imp_rate_ra  = np.zeros(num_implants, dtype=np.float32)
    imp_rate_dec = np.zeros(num_implants, dtype=np.float32)
    imp_ref      = np.zeros((num_implants, 2), dtype=np.float32)
    positions    = np.zeros((num_implants, T, 2), dtype=np.float32)
    neg_positions = np.zeros((num_implants, T, n_tmpl, 2), dtype=np.float32)
    fluxes       = np.zeros((num_implants, T), dtype=np.float32)

    for k in range(num_implants):

        # resample orbit+position until the whole track fits in every frame
        for _ in range(MAX_ORBIT_TRIES):
            # fake orbital paramters
            orbit = orbit_sampler.sample()
            # paramters to apparent sky motion
            motion = MotionModel.compute(orbit)

            ref = sample_visible_reference_position(
                rng, motion, affines, cent_times, cent_time0,
                pscales, H, W, pad_x, pad_y)
            if ref is not None:
                break
        else:
            raise RuntimeError(
                "Could not find an implant position visible in every frame")

        u_ref, v_ref = ref

        # random magnitude
        mag = mag_sampler.sample()

        # template line PSFs only depend on this implant's motion,
        # build them once instead of once per frame
        if add_negative_wells:
            tmpl_mpsfs = [
                build_line_psf(tmpl_psf_files[j],
                               motion["rate"],
                               motion["rate_ra"], motion["rate_dec"],
                               tmpl_expt[j], tmpl_pscale[j])
                for j in range(n_tmpl)
            ]

        for i in range(T):
            # map the frame-0 crop position into this frame's crop grid
            a = affines[i]
            x_i = a[0, 0] * u_ref + a[0, 1] * v_ref + a[0, 2]
            y_i = a[1, 0] * u_ref + a[1, 1] * v_ref + a[1, 2]

            # Compute time difference from frame 0
            dt_hr = (cent_times[i] - cent_time0) * 24.0

            # expected pixel displacement from motion
            x_inj = x_i + motion["rate_ra"] * dt_hr / pscales[i]
            y_inj = y_i - motion["rate_dec"] * dt_hr / pscales[i]

            # positive source: counts from the magnitude/zeropoint
            pos_counts = PhotoCalib.mag_to_counts(mag, zps[i])
            mpsf = build_line_psf(psf_files[i],
                                  motion["rate"],
                                  motion["rate_ra"], motion["rate_dec"],
                                  exptimes[i], pscales[i])
            model = make_psf_image(mpsf, (H, W), x_inj, y_inj, pos_counts)

            # noiseless positive model on the reference grid
            if track_target is not None:
                x_t = u_ref + motion["rate_ra"] * dt_hr / ref_pscale
                y_t = v_ref - motion["rate_dec"] * dt_hr / ref_pscale
                mpsf_ref = build_line_psf(psf_files[i],
                                          motion["rate"],
                                          motion["rate_ra"], motion["rate_dec"],
                                          exptimes[i], ref_pscale)
                tgt_model = make_psf_image(mpsf_ref, (H, W), x_t, y_t, pos_counts)
                track_target += tgt_model.astype(np.float32)

            # variance goes up by the Poisson contribution of the model
            variance[i] += (np.abs(model) / gains[i]).astype(np.float32)
            fluxes[k, i] = float(np.sum(model))

            if add_poisson_noise:
                # photon noise drawn from the same rng, deterministic
                model = model + rng.standard_normal((H, W)) * np.sqrt(
                    np.abs(model) / gains[i])

            science[i] += model.astype(np.float32)

            # negative wells: where the object sat during each template
            for j in range(n_tmpl):
                dt_j = (tmpl_ct[j] - cent_time0) * 24.0
                xj = x_i + motion["rate_ra"] * dt_j / pscales[i]
                yj = y_i - motion["rate_dec"] * dt_j / pscales[i]
                neg_counts = -PhotoCalib.mag_to_counts(mag, tmpl_zp[j]) / n_tmpl
                well = make_psf_image(tmpl_mpsfs[j], (H, W), xj, yj, neg_counts)
                science[i] += well.astype(np.float32)
                neg_positions[k, i, j] = (xj, yj)

            positions[k, i] = (x_inj, y_inj)

        imp_r[k] = orbit["r"]
        imp_e[k] = orbit["e"]
        imp_i_deg[k] = orbit["i_deg"]
        imp_mag[k] = mag
        imp_rate[k] = motion["rate"]
        imp_angle[k] = motion["angle"]
        imp_rate_ra[k] = motion["rate_ra"]
        imp_rate_dec[k] = motion["rate_dec"]
        imp_ref[k] = (u_ref, v_ref)

    # optional stack over the injected sequence
    if output_mode == "mean":
        final_stack = np.mean(science, axis=0).astype(np.float32)
    elif output_mode == "median":
        final_stack = np.median(science, axis=0).astype(np.float32)
    else:
        final_stack = None

    # keep the label shapes fixed, default collate chokes on None
    has_final = final_stack is not None
    has_track = track_target is not None

    # labels
    labels = {
        "num_implants": np.int64(num_implants),
        "n_template": np.int64(n_tmpl),
        "r": imp_r,
        "e": imp_e,
        "i_deg": imp_i_deg,
        "mag": imp_mag,
        "rate": imp_rate,
        "angle": imp_angle,
        "rate_ra": imp_rate_ra,
        "rate_dec": imp_rate_dec,
        "ref_pos_crop": imp_ref,
        "positions_crop": positions,
        "negative_positions_crop": neg_positions,
        "injected_fluxes": fluxes,
        "output_mode": output_mode,
        "final_stack": final_stack if has_final else np.zeros((H, W), dtype=np.float32),
        "track_target": track_target if has_track else np.zeros((H, W), dtype=np.float32),
        "has_final_stack": np.bool_(has_final),
        "has_track_target": np.bool_(has_track),
    }

    return science, variance, mask, labels
