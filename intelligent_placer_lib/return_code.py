from enum import Enum, IntEnum


class RCWho(str, Enum):
    unknown = 'Unknown'
    engine = 'Engine'
    packer = 'Packer'
    recognition = 'Recognition'


class RCType(IntEnum):
    UNKNOWN = 0
    SUCCESS = 1
    CUSTOM_ERROR = 2
    NULL_ARGUMENT = 3
    FILE_NOT_FOUND = 4


class RC:
    class ReturnCode:
        def __init__(self, rc_who: RCWho, rc_type: RCType, info: str):
            self.rc_who = rc_who
            self.rc_type = rc_type
            self.info = info

        def __str__(self):
            return 'ERROR:\n\tfrom: ' + str(self.rc_who.value) \
                   + '\n\ttype: ' + str(self.rc_type)[len('RCType.'):] + '\n\tmessage: ' \
                   + str(self.info)

    @staticmethod
    def is_success(code: ReturnCode) -> bool:
        return code.rc_type == RCType.SUCCESS

    RC_SUCCESS = ReturnCode(RCWho.unknown, RCType.SUCCESS, "No errors")
    RC_ENGINE_FILE_ERROR = ReturnCode(RCWho.engine, RCType.FILE_NOT_FOUND, "image path is empty")
