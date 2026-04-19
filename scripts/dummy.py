#!/usr/bin/env python3

from fastgs_core import dummy_add


def main() -> None:
    result = dummy_add(20, 22)
    print(f"dummy_add(20, 22) = {result}")


if __name__ == "__main__":
    main()
