import argparse
import io
from pathlib import Path

import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
import zstandard as zstd
from torch.utils.data import DataLoader, IterableDataset


BASE_MOVE_CLASSES = 64 * 64
PROMOTION_MOVE_CLASSES = 2 * 8 * 3 * 4
NUM_MOVE_CLASSES = BASE_MOVE_CLASSES + PROMOTION_MOVE_CLASSES

PROMOTION_TO_INDEX = {
    chess.KNIGHT: 0,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 3,
}


class ChessAI(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, NUM_MOVE_CLASSES)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        x = self.flatten(x)
        x = torch.relu(self.fc1(x))

        return self.fc2(x)


def board_to_tensor(board):
    tensor = torch.zeros((12, 8, 8), dtype=torch.float32)

    piece_to_layer = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if piece is None:
            continue

        row = chess.square_rank(square)
        col = chess.square_file(square)

        layer = piece_to_layer[piece.piece_type]

        if piece.color == chess.BLACK:
            layer += 6

        tensor[layer][row][col] = 1.0

    return tensor


def move_to_index(move):
    if move.promotion is None:
        return move.from_square * 64 + move.to_square

    if move.promotion not in PROMOTION_TO_INDEX:
        raise ValueError(f"Unsupported promotion piece: {move.promotion}")

    from_rank = chess.square_rank(move.from_square)
    from_file = chess.square_file(move.from_square)
    to_file = chess.square_file(move.to_square)

    if from_rank == 6:
        color_index = 0
    elif from_rank == 1:
        color_index = 1
    else:
        raise ValueError(f"Invalid promotion move: {move.uci()}")

    file_delta = to_file - from_file

    if file_delta not in (-1, 0, 1):
        raise ValueError(f"Invalid promotion move: {move.uci()}")

    direction_index = file_delta + 1
    promotion_index = PROMOTION_TO_INDEX[move.promotion]

    offset = (
        ((color_index * 8 + from_file) * 3 + direction_index) * 4
        + promotion_index
    )

    return BASE_MOVE_CLASSES + offset


class ChessStreamDataset(IterableDataset):
    def __init__(self, file_path):
        self.file_path = Path(file_path)

    def __iter__(self):
        with self.file_path.open("rb") as compressed_file:
            decompressor = zstd.ZstdDecompressor()

            with decompressor.stream_reader(compressed_file) as reader:
                text_stream = io.TextIOWrapper(
                    reader,
                    encoding="utf-8",
                    errors="replace",
                )

                while True:
                    game = chess.pgn.read_game(text_stream)

                    if game is None:
                        break

                    board = game.board()

                    for move in game.mainline_moves():
                        if not board.is_legal(move):
                            break

                        position = board_to_tensor(board)
                        target = move_to_index(move)

                        yield position, target
                        board.push(move)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        print("No checkpoint found. Starting new training.")
        return

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {checkpoint_path}")


def save_checkpoint(model, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    torch.save(model.state_dict(), checkpoint_path)


def train(dataset_path, checkpoint_path, batch_size, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    print(f"Using device: {device}")

    model = ChessAI().to(device)

    try:
        load_checkpoint(model, checkpoint_path, device)
    except RuntimeError as error:
        print(f"Checkpoint is incompatible with the current model: {error}")
        print("Starting with a new model.")
        model = ChessAI().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_cuda,
    )

    dataset = ChessStreamDataset(dataset_path)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
    )

    positions_trained = 0
    batch_index = 0

    print("Starting training...")

    try:
        for positions, targets in data_loader:
            positions = positions.to(device, non_blocking=use_cuda)
            targets = targets.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_cuda,
            ):
                predictions = model(positions)
                loss = criterion(predictions, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            positions_trained += len(positions)
            batch_index += 1

            if batch_index % 10 == 0:
                print(
                    f"Batch {batch_index} | "
                    f"Positions: {positions_trained} | "
                    f"Loss: {loss.item():.4f}"
                )

            if batch_index % 100 == 0:
                save_checkpoint(model, checkpoint_path)
                print(
                    f"Checkpoint saved after "
                    f"{positions_trained} positions"
                )

    except KeyboardInterrupt:
        print("\nTraining stopped by user.")

    finally:
        if batch_index > 0:
            save_checkpoint(model, checkpoint_path)
            print(f"Final checkpoint saved: {checkpoint_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a chess move prediction model from compressed PGN data."
    )

    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to a .pgn.zst file",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("chess_model_highspeed.pth"),
        help="Path used to load and save the model checkpoint",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Training batch size",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Optimizer learning rate",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if not args.dataset.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    if args.batch_size <= 0:
        raise ValueError("Batch size must be greater than 0.")

    if args.learning_rate <= 0:
        raise ValueError("Learning rate must be greater than 0.")

    train(
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
