class RiseMachineSpiritError(Exception):
    """Base exception for user-facing failures."""


class UnsupportedCarrierError(RiseMachineSpiritError):
    """Raised when the audio carrier is lossy or unsupported."""


class CapacityError(RiseMachineSpiritError):
    """Raised when the payload does not fit in the carrier."""

    def __init__(self, capacity_bytes: int, required_bytes: int):
        self.capacity_bytes = capacity_bytes
        self.required_bytes = required_bytes
        overflow = max(0, required_bytes - capacity_bytes)
        super().__init__(
            f"Payload requires {required_bytes} bytes but carrier capacity is "
            f"{capacity_bytes} bytes; overflow is {overflow} bytes."
        )


class PayloadFormatError(RiseMachineSpiritError):
    """Raised when an extracted payload header is invalid."""


class AuthenticationError(RiseMachineSpiritError):
    """Raised when recovered payload bytes fail authenticated decryption."""

    def __init__(
        self,
        expected_hint: int,
        actual_hint: int,
        suspect_frames: list[str] | None = None,
    ):
        self.expected_hint = expected_hint
        self.actual_hint = actual_hint
        self.suspect_frames = suspect_frames or []
        suffix = ""
        if self.suspect_frames:
            suffix = " Suspect frame/channel slots: " + ", ".join(self.suspect_frames)
        super().__init__(
            "Payload authentication failed; wrong key, wrong parameters, or corrupted "
            f"carrier. Hints: 0x{expected_hint:08x}/0x{actual_hint:08x}." + suffix
        )


CrcMismatchError = AuthenticationError
