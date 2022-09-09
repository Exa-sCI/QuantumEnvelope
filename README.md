[![Build Status](https://travis-ci.org/TApplencourt/QuantumEnvelope.svg?branch=master)](https://travis-ci.org/TApplencourt/QuantumEnvelope)


# How to run the tests


```
./tests/test_everything_all_at_once.py
python -m doctest -o NORMALIZE_WHITESPACE -v */*.py
```


# How to pass the linter

```
black --line-length 100 */*.py
```
