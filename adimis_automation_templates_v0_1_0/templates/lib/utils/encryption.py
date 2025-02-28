import os
import json
from typing import TypeVar, Generic
from base64 import b64encode, b64decode
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.padding import PKCS7
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

T = TypeVar("T")


class EncryptionClient(Generic[T]):
    def __init__(self, key: str):
        self.key = key.encode("utf-8")
        """Initialize the encryption class with a key."""
        if not key:
            raise ValueError("Encryption key cannot be empty.")
        self.key = key

    def _derive_key(self, key: bytes, salt: bytes) -> bytes:
        """Derives a strong key from the provided key using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend(),
        )
        return kdf.derive(key)

    def _get_cipher(self, key: bytes, iv: bytes) -> Cipher:
        """Create a cipher object with the given key and IV."""
        return Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())

    def encrypt(self, data: T) -> str:
        """Encrypts data of any type."""
        if isinstance(data, (dict, list)):
            data = json.dumps(data)
        elif not isinstance(data, str):
            data = str(data)

        data_bytes = data.encode("utf-8")
        salt = os.urandom(16)
        iv = os.urandom(16)
        key = self._derive_key(self.key, salt)
        cipher = self._get_cipher(key, iv)

        padder = PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(data_bytes) + padder.finalize()
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

        return b64encode(salt + iv + encrypted_data).decode("utf-8")

    def decrypt(self, encrypted_data: str) -> T:
        """Decrypts data and tries to reconstruct the original type."""
        encrypted_data_bytes = b64decode(encrypted_data)

        salt = encrypted_data_bytes[:16]
        iv = encrypted_data_bytes[16:32]
        encrypted_content = encrypted_data_bytes[32:]

        key = self._derive_key(self.key, salt)
        cipher = self._get_cipher(key, iv)

        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encrypted_content) + decryptor.finalize()
        unpadder = PKCS7(algorithms.AES.block_size).unpadder()
        data_bytes = unpadder.update(padded_data) + unpadder.finalize()
        data_str = data_bytes.decode("utf-8")

        try:
            return json.loads(data_str)
        except json.JSONDecodeError:
            try:
                return float(data_str) if "." in data_str else int(data_str)
            except ValueError:
                return data_str
