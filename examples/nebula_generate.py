import argparse
import json

from nebula import DecoderConfig, Engine, ParallelContext


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--tokens", type=int, default=4)
    args = parser.parse_args()

    context = ParallelContext.from_env(tp_size=args.tp, pp_size=args.pp, device=args.device)
    engine = None
    try:
        engine = Engine(DecoderConfig(), context=context)
        if context.rank == 0:
            out = engine.generate([[1, 2, 3], [4, 5, 6]], args.tokens)
            print(json.dumps({"generated": out}))
        else:
            engine.serve()
    finally:
        if engine is not None:
            engine.close()
        context.close()


if __name__ == "__main__":
    main()
