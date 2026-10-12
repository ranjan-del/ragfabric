# Source before running the benchmark scripts. Override any value in your shell first.
# The database is a local throwaway PostgreSQL with pgvector; nothing here is a real secret.
export RFB_BENCH_DIR="${RFB_BENCH_DIR:-$HOME/ragfabric-bench}"
export DATABASE_URL="${DATABASE_URL:-postgresql+psycopg://ragfabric:ragfabric@localhost:5432/ragfabric_bench}"
export JWT_SECRET="${JWT_SECRET:-local-benchmark-only}"
export RFB_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/configs"
