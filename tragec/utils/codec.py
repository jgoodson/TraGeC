"""
Copyright (c) 2010, Kou Man Tong
Copyright (c) 2015, Ayun Park
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Kou Man Tong nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import struct
from datetime import datetime, timezone
from abc import ABCMeta, abstractmethod
from uuid import UUID

from binascii import b2a_hex

utc = timezone.utc


class MissingClassDefinition(ValueError):
    def __init__(self, class_name):
        super(MissingClassDefinition,
              self).__init__("No class definition for class %s" % (class_name,))


class UnknownSerializerError(ValueError):
    def __init__(self, key, value):
        super(UnknownSerializerError,
              self).__init__("Unable to serialize: key '%s' value: %s type: %s" % (key, value, type(value)))


class MissingTimezoneWarning(RuntimeWarning):
    def __init__(self, *args):
        args = list(args)
        if len(args) < 1:
            args.append("Input datetime object has no tzinfo, assuming UTC.")
        super(MissingTimezoneWarning, self).__init__(*args)


class TraversalStep(object):
    def __init__(self, parent, key):
        self.parent = parent
        self.key = key


class BSONCoding(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def bson_encode(self):
        pass

    @abstractmethod
    def bson_init(self, raw_values):
        pass


classes = {}


def import_class(cls):
    if not issubclass(cls, BSONCoding):
        return

    global classes
    classes[cls.__name__] = cls


def import_classes(*args):
    for cls in args:
        import_class(cls)


def import_classes_from_modules(*args):
    for module in args:
        for item in module.__dict__:
            if hasattr(item, "__new__") and hasattr(item, "__name__"):
                import_class(item)


def encode_object(obj, traversal_stack, generator_func, on_unknown=None):
    values = obj.bson_encode()
    class_name = obj.__class__.__name__
    values["$$__CLASS_NAME__$$"] = class_name
    return encode_document(values, traversal_stack, obj,
                           generator_func, on_unknown)


def encode_object_element(name, value, traversal_stack,
                          generator_func, on_unknown):
    return b"\x03" + encode_cstring(name) + \
           encode_object(value, traversal_stack,
                         generator_func=generator_func, on_unknown=on_unknown)


class _EmptyClass(object):
    pass


def decode_object(raw_values):
    global classes
    class_name = raw_values["$$__CLASS_NAME__$$"]
    try:
        cls = classes[class_name]
    except KeyError:
        raise MissingClassDefinition(class_name)

    retval = _EmptyClass()
    retval.__class__ = cls
    alt_retval = retval.bson_init(raw_values)
    return alt_retval or retval


def encode_string(value):
    value = value.encode("utf-8")
    length = len(value)
    return struct.pack("<i%dsb" % (length,), length + 1, value, 0)


def encode_cstring(value):
    if not isinstance(value, bytes):
        value = str(value).encode("utf-8")
    if b"\x00" in value:
        raise ValueError("Element names may not include NUL bytes.")
        # A NUL byte is used to delimit our string, accepting one would cause
        # our string to terminate early.
    return value + b"\x00"


def encode_binary(value, binary_subtype=0):
    length = len(value)
    return struct.pack("<ib", length, binary_subtype) + value


def encode_double(value):
    return struct.pack("<d", value)


ELEMENT_TYPES = {
    0x01: "double",
    0x02: "string",
    0x03: "document",
    0x04: "array",
    0x05: "binary",
    0x07: "object_id",
    0x08: "boolean",
    0x09: "UTCdatetime",
    0x0A: "none",
    0x10: "int32",
    0x11: "uint64",
    0x12: "int64"
}


def encode_double_element(name, value):
    return b"\x01" + encode_cstring(name) + encode_double(value)


def encode_string_element(name, value):
    return b"\x02" + encode_cstring(name) + encode_string(value)


def _is_string(value):
    if isinstance(value, str):
        return True
    return False


def decode_binary_subtype(value, binary_subtype):
    if binary_subtype in [0x03, 0x04]:  # legacy UUID, UUID
        return UUID(bytes=value)
    return value


def mvindex(data, match, start=None, end=None):
    for idx in range(start if start else 0, end if end else len(data)):
        if data[idx] == match:
            break
    return idx


def decode_document(data, base, as_array=False):
    # Create all the struct formats we might use.
    double_struct = struct.Struct("<d")
    int_struct = struct.Struct("<i")
    char_struct = struct.Struct("<b")
    long_struct = struct.Struct("<q")
    uint64_struct = struct.Struct("<Q")
    int_char_struct = struct.Struct("<ib")

    length = struct.unpack("<i", data[base:base + 4])[0]
    end_point = base + length
    if data[end_point - 1] not in ('\0', 0):
        raise ValueError('missing null-terminator in document')
    base += 4
    retval = [] if as_array else {}
    decode_name = not as_array

    while base < end_point - 1:

        element_type = char_struct.unpack(data[base:base + 1])[0]

        ll = mvindex(data, 0, base + 1) + 1
        if decode_name:
            name = data[base + 1:ll - 1]
            try:
                name = name.decode("utf-8")
            except UnicodeDecodeError:
                pass
        else:
            name = None
        base = ll

        if element_type == 0x01:  # double
            value = double_struct.unpack(data[base: base + 8])[0]
            base += 8
        elif element_type == 0x02:  # string
            length = int_struct.unpack(data[base:base + 4])[0]
            value = data[base + 4: base + 4 + length - 1]
            value = value.decode("utf-8")
            base += 4 + length
        elif element_type == 0x03:  # document
            base, value = decode_document(data, base)
        elif element_type == 0x04:  # array
            base, value = decode_document(data, base, as_array=True)
        elif element_type == 0x05:  # binary
            length, binary_subtype = int_char_struct.unpack(
                data[base:base + 5])
            value = data[base + 5:base + 5 + length]
            value = decode_binary_subtype(value, binary_subtype)
            base += 5 + length
        elif element_type == 0x07:  # object_id
            value = b2a_hex(data[base:base + 12])
            base += 12
        elif element_type == 0x08:  # boolean
            value = bool(char_struct.unpack(data[base:base + 1])[0])
            base += 1
        elif element_type == 0x09:  # UTCdatetime
            value = datetime.fromtimestamp(
                long_struct.unpack(data[base:base + 8])[0] / 1000.0, utc)
            base += 8
        elif element_type == 0x0A:  # none
            value = None
        elif element_type == 0x10:  # int32
            value = int_struct.unpack(data[base:base + 4])[0]
            base += 4
        elif element_type == 0x11:  # uint64
            value = uint64_struct.unpack(data[base:base + 8])[0]
            base += 8
        elif element_type == 0x12:  # int64
            value = long_struct.unpack(data[base:base + 8])[0]
            base += 8

        if as_array:
            retval.append(value)
        else:
            retval[name] = value
    if "$$__CLASS_NAME__$$" in retval:
        retval = decode_object(retval)
    return end_point, retval
