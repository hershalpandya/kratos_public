# Sample file for unittesting with PyTest
# The filename should be of the form test_*.py or *_test.py

# Some random function to test -> will probably be imported in the general case
def func(x):
    return x + 1

# The first option is to write a function prefixed by 'test'
def test_answer():
    assert func(3) == 5

# The second option is to make a class prefixed by 'Test', with 'test' prefixed functions
class TestClass:
    def test_one(self):
        x = "this"
        assert "h" in x

    def test_two(self):
        x = "hello"
        assert hasattr(x, "check")
