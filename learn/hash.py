import hashlib

text = "Return policy is 30 days."

hashed = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

print(hashed)
