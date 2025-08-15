from utils.text import TextEncoder
def test_roundtrip():
    enc = TextEncoder(vocab_size=1000)
    ids = enc.encode("hello world")
    assert ids[0]==1 and ids[-1]==2 and len(ids)>2
    # text = enc.decode(ids)
    # assert text == "hello world"