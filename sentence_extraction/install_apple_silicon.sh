#!/bin/bash
curl https://sh.rustup.rs -sSf | sh
export PATH="$HOME/.cargo/bin:$PATH"
export RUSTFLAGS="-A invalid_reference_casting"

pip install -r requirements.txt