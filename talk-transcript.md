Start the BERT server

```
./star-bert.sh
```

Enter the virtualenv

```
dlenv
```

Warmup the BERT server

```
python3 warmup.py
```

Train the models:

```
python3 train.py data/
```

Download the movie script:

```
python3 down_parse_subtitle.py data/
```

Evaluate the models:

```
python3 evaluate.py data/
```
