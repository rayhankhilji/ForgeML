import asyncio
import json

from nebula import AsyncBatcher, DecoderConfig, Engine, RequestRouter


async def main() -> None:
    config = DecoderConfig(
        vocab_size=32,
        hidden_size=16,
        num_heads=4,
        num_layers=4,
        intermediate_size=32,
        max_seq_len=32,
        seed=17,
    )
    engines = [Engine(config), Engine(config)]
    batchers = [AsyncBatcher(e, max_batch_size=4) for e in engines]
    router = RequestRouter(batchers)
    try:
        results = await asyncio.gather(
            *[router.generate([i + 1, i + 2, i + 3], 4) for i in range(8)]
        )
        print(json.dumps({"generated": results}))
    finally:
        await router.close()


if __name__ == "__main__":
    asyncio.run(main())
