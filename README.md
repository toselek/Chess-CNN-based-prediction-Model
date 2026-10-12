# Chess-CNN-based-prediction-Model
PyTorch CNN for predicting chess moves from streamed Lichess PGN data
# Chess Move Prediction with PyTorch

A small machine learning project that trains a convolutional neural network to predict chess moves from board positions.

The training data is read directly from compressed Lichess PGN files, so large datasets do not need to be extracted before training.

## Model

The board is represented as a `12 x 8 x 8` tensor.

There are six piece planes for White and six for Black:

- Pawn
- Knight
- Bishop
- Rook
- Queen
- King

The neural network consists of three convolutional layers followed by two fully connected layers.

Move labels are encoded using the source and destination squares. Promotion moves are represented separately so that promotions to a knight, bishop, rook, or queen can be distinguished.

## Requirements

- Python 3
- PyTorch
- python-chess
- zstandard

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset

The script expects a compressed PGN file in `.pgn.zst` format.

Lichess provides public chess game databases that can be used for training.

The dataset itself is not included in this repository.

## Training

Run:

```bash
python train.py path/to/games.pgn.zst
```

Optional arguments:

```bash
python train.py path/to/games.pgn.zst \
    --batch-size 1024 \
    --learning-rate 0.001 \
    --checkpoint chess_model_highspeed.pth
```

CUDA is used automatically when a compatible GPU is available.

The model is saved periodically during training and is also saved when training is stopped with `Ctrl+C`.

## Checkpoints

By default, model weights are stored in:

```text
chess_model_highspeed.pth
```

If a compatible checkpoint already exists, its weights are loaded before training starts.

Model checkpoints and datasets are excluded from the repository through `.gitignore`.

## Current limitations

The current board representation only stores piece locations. It does not yet encode:

- side to move
- castling rights
- en passant state
- move history

Because of this, two chess positions with identical piece placement but different game state can currently receive the same input representation.

The model is trained to predict moves played in the dataset. It should therefore be considered a move-prediction model rather than a chess engine that searches for the objectively best move.

## Future improvements

Possible next steps include:

- adding side-to-move information
- encoding castling and en passant rights
- adding validation data and accuracy metrics
- filtering games by player rating
- improving the neural network architecture
- comparing predictions with engine evaluations
- adding inference code to predict moves from a FEN position
