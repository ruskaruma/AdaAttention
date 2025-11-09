from cuda_attn._ext import load_extension


def main():
    load_extension(rebuild=True)

if __name__ == "__main__":
    main()

