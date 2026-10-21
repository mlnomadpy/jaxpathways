"""Frozen pilot interfaces for the C04 raw-audio ASR work."""

SAMPLE_RATE = 16_000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 80
FMIN_HZ = 20.0
FMAX_HZ = 7_600.0
LOG_FLOOR = 1e-5
MAX_SECONDS = 8.0
MIN_SECONDS = 0.5
MAX_SAMPLES = int(MAX_SECONDS * SAMPLE_RATE)

# CTC blank is deliberately separate from the printable character inventory.
TOKENS = ("<blank>", " ", "'", *tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
BLANK_ID = 0
TOKEN_TO_ID = {token: i for i, token in enumerate(TOKENS)}
ID_TO_TOKEN = dict(enumerate(TOKENS))

PILOT_MODEL = {
    "frontend_channels": 32,
    "frontend_kernel": 3,
    "frontend_stride": 2,
    "width": 32,
    "blocks": 2,
    "heads": 4,
    "mlp_width": 64,
    "dropout": 0.1,
}
