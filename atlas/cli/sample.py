"""``atlas-sample`` — generate text from a checkpoint."""

from __future__ import annotations

import argparse

from atlas.inference.generate import generate_from_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description="Atlas: generate text from checkpoint")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    p.add_argument("--prompt", type=str, default="", help="Text prompt")
    p.add_argument("--tokens", type=int, default=200, help="Max tokens to generate")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--tokenizer-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    text = generate_from_checkpoint(
        ckpt_path=args.ckpt,
        prompt=args.prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        tokenizer_dir=args.tokenizer_dir,
        device=args.device,
    )
    print(text)


if __name__ == "__main__":
    main()
