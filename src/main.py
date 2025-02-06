from src.data_preprocessing import load_data
from src.nmf import train_nmf


def main():
    V, movie_info = load_data()
    W, H = train_nmf(V, 10, 100, 1e-5)

if __name__ == "__main__":
    main()