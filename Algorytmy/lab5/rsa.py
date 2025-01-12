import math
from sympy import mod_inverse

def generate_keys(p, q):
    n = p * q

    phi = (p - 1) * (q - 1)

    e = 3
    while math.gcd(e, phi) != 1:
        e += 1

    d = mod_inverse(e, phi)

    return (n, e), (n, d)

def encrypt(public_key, plaintext):
    n, e = public_key

    plaintext_ascii = [ord(char) for char in plaintext]

    ciphertext = [pow(char, e, n) for char in plaintext_ascii]
    return ciphertext


def decrypt(private_key, ciphertext):
    n, d = private_key

    decrypted_ascii = [pow(char, d, n) for char in ciphertext]

    plaintext = ''.join(chr(char) for char in decrypted_ascii)
    return plaintext


if __name__ == "__main__":

    p = 1061
    q = 1063
    plaintext = "dom"

    print("Tekst do zaszyfrowania:", plaintext)

    public_key, private_key = generate_keys(p, q)

    print("Klucz publiczny:", public_key)
    print("Klucz prywatny:", private_key)




    ciphertext = encrypt(public_key, plaintext)
    print("Szyfrogram:", ciphertext)


    decrypted_text = decrypt(private_key, ciphertext)
    print("Odszyfrowany tekst:", decrypted_text)