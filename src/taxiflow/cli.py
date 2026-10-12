"""Typer CLI interface for Taxiflow."""

import typer
import uvicorn
from rich.console import Console

from taxiflow import __version__

app = typer.Typer(help="Taxiflow: NYC Taxi Fare Prediction Under Pricing Drift.")
console = Console()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host address to bind to"),
    port: int = typer.Option(8000, help="Port to listen on"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
):
    """Start the Taxiflow FastAPI prediction service."""
    console.print(f"[bold green]Starting Taxiflow API on http://{host}:{port}[/bold green]")
    uvicorn.run("taxiflow.serving.api:app", host=host, port=port, reload=reload)


@app.command()
def download(
    start: str = typer.Option("2024-01", help="First month (YYYY-MM)"),
    end: str = typer.Option("2026-08", help="Last month (YYYY-MM), inclusive"),
    raw_dir: str = typer.Option("data/raw", help="Destination directory"),
):
    """Download monthly TLC Yellow taxi parquet files."""
    from taxiflow.data.download import download_yellow

    paths = download_yellow(start, end, raw_dir)
    console.print(f"[bold green]{len(paths)} monthly files ready in {raw_dir}[/bold green]")


@app.command()
def clean(
    config: str = typer.Option("configs/default.yaml", help="Path to config file"),
    force: bool = typer.Option(False, help="Re-clean months that already have output"),
):
    """Filter raw monthly files into compact cleaned parquet (one per month)."""
    from taxiflow.config import AppConfig
    from taxiflow.data.clean import clean_all
    from taxiflow.data.download import month_range

    cfg = AppConfig.from_yaml(config)
    months = month_range(cfg.data.start_month, cfg.data.end_month)
    report = clean_all(months, cfg.data.raw_dir, cfg.data.processed_dir, cfg.data.filters,
                       cfg.data.known_bad_records, force)
    n_raw = sum(r["raw_rows"] for r in report)
    n_clean = sum(r["clean_rows"] for r in report)
    console.print(f"[bold green]{n_raw:,} raw trips -> {n_clean:,} clean ({n_clean / n_raw:.1%})[/bold green]")


@app.command()
def train(config: str = typer.Option("configs/default.yaml", help="Path to config file")):
    """Train the main model, the 2024 drift-study model and the baselines."""
    from taxiflow.config import AppConfig
    from taxiflow.pipeline import run_train

    run_train(AppConfig.from_yaml(config), log=console.print)


@app.command()
def evaluate(config: str = typer.Option("configs/default.yaml", help="Path to config file")):
    """Evaluate all models on the out-of-time test months and write reports/metrics.json."""
    from taxiflow.config import AppConfig
    from taxiflow.pipeline import run_evaluate

    run_evaluate(AppConfig.from_yaml(config), log=console.print)


@app.command()
def walkforward(config: str = typer.Option("configs/default.yaml", help="Path to config file")):
    """Compare static / monthly bias-corrected / monthly retrained models; writes reports/walkforward.json."""
    import json
    from pathlib import Path

    from taxiflow.config import AppConfig
    from taxiflow.evaluation.walkforward import run_walkforward
    from taxiflow.models.train import FareModel
    from taxiflow.pipeline import BASE_MODEL_PATH

    cfg = AppConfig.from_yaml(config)
    results = run_walkforward(cfg, FareModel.load(BASE_MODEL_PATH), log=console.print)
    out = Path("reports/walkforward.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    console.print(f"[bold green]wrote {out}[/bold green]")


@app.command()
def correct(
    month: str = typer.Option(..., help="Most recent complete month of trips (YYYY-MM)"),
    config: str = typer.Option("configs/default.yaml", help="Path to config file"),
):
    """Apply last month's per-booking-type bias correction to the trained model for serving."""
    from taxiflow.config import AppConfig
    from taxiflow.models.train import FareModel, features_for, load_window
    from taxiflow.pipeline import BASE_MODEL_PATH

    cfg = AppConfig.from_yaml(config)
    model = FareModel.load(BASE_MODEL_PATH)
    df = load_window(cfg.data.processed_dir, (month, month))
    offsets = model.apply_monthly_correction(features_for(df, model.routes), df["target"].to_numpy(), month, cfg.seed)
    model.save(cfg.serving.model_path)
    console.print(f"bias offsets from {month}: metered {offsets['metered']:+.2f}, flex fare {offsets['flex_fare']:+.2f}")
    console.print(f"[bold green]saved serving model to {cfg.serving.model_path}[/bold green]")


@app.command()
def info():
    """Print project version and configuration overview."""
    console.print(f"[bold blue]Taxiflow version {__version__}[/bold blue]")
    console.print("NYC Taxi Fare Prediction Under Pricing Drift, with conformal prediction intervals.")


if __name__ == "__main__":
    app()
