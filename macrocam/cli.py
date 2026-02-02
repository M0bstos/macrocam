from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from macrocam.cache import get_vision_cache, set_vision_cache
from macrocam.nutrition import lookup_usda_food
from macrocam.utils import (
    SUPPORTED_IMAGE_EXTENSIONS,
    is_supported_image,
    parse_grams,
    require_existing_file,
    sha256_file,
)
from macrocam.vision import analyze_image_json, normalize_candidates


app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _print_header() -> None:
    title = "[bold]MacroCam[/bold]  photo -> macros (estimated)"
    console.print(Panel.fit(title, border_style="bold green"))
    console.print(
        "[dim]Disclaimer:[/dim] This tool provides estimates based on database matches."
    )


def _get_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in MIME_TYPES:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise ValueError(f"Unsupported image extension: {ext}. Supported: {supported}")
    return MIME_TYPES[ext]


def _prompt_label(candidates) -> str:
    table = Table(title="Detected foods", header_style="bold magenta")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Label", style="bold")
    table.add_column("Confidence", justify="right")
    table.add_column("Notes", style="dim")
    for idx, item in enumerate(candidates, start=1):
        notes = item.notes if item.notes else "-"
        table.add_row(str(idx), item.label, f"{item.confidence:.2f}", notes)
    console.print(table)

    prompt = f"Select 1-{len(candidates)} or type your own"
    while True:
        choice = Prompt.ask(prompt, default="1").strip()
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(candidates):
                return candidates[index - 1].label
        if choice:
            return choice
        console.print("[red]Please choose a number or enter a food label.[/red]")


def _prompt_grams() -> float:
    while True:
        raw = Prompt.ask("Portion (grams)", default="100")
        try:
            return parse_grams(raw)
        except ValueError as exc:
            console.print(f"[red]Invalid grams:[/red] {exc}")


def _scale_optional(value: float | None, multiplier: float) -> float | None:
    if value is None:
        return None
    return value * multiplier


def _render_macros(grams: float, match) -> None:
    multiplier = grams / 100.0
    facts = match.per_100g
    calories = facts.calories_kcal * multiplier
    protein = facts.protein_g * multiplier
    carbs = facts.carbs_g * multiplier
    fat = facts.fat_g * multiplier

    table = Table(title=f"Macros ({grams:.0f}g)")
    table.add_column("Nutrient", style="bold")
    table.add_column("Amount", justify="right")
    table.add_row("Calories", f"{calories:.0f} kcal")
    table.add_row("Protein", f"{protein:.1f} g")
    table.add_row("Carbs", f"{carbs:.1f} g")
    table.add_row("Fat", f"{fat:.1f} g")

    sugars = _scale_optional(facts.sugars_g, multiplier)
    fiber = _scale_optional(facts.fiber_g, multiplier)
    sodium = _scale_optional(facts.sodium_mg, multiplier)
    cholesterol = _scale_optional(facts.cholesterol_mg, multiplier)

    if sugars is not None:
        table.add_row("Sugars", f"{sugars:.1f} g")
    if fiber is not None:
        table.add_row("Fiber", f"{fiber:.1f} g")
    if sodium is not None:
        table.add_row("Sodium", f"{sodium:.0f} mg")
    if cholesterol is not None:
        table.add_row("Cholesterol", f"{cholesterol:.0f} mg")

    console.print(table)


@app.command()
def main(
    image_path: Path = typer.Argument(..., help="Path to a food image"),
    out: Path | None = typer.Option(None, "--out", help="Output label path (phase 7)"),
    no_interactive: bool = typer.Option(
        False, "--no-interactive", help="Disable prompts (requires --food and --grams)"
    ),
    food: str | None = typer.Option(None, "--food", help="Food label override"),
    grams: str | None = typer.Option(None, "--grams", help="Portion size, grams"),
    data_dir: Path | None = typer.Option(None, "--data-dir", help="USDA CSV directory"),
) -> None:
    _print_header()

    try:
        image_file = require_existing_file(image_path)
        if not is_supported_image(image_file):
            raise ValueError("Unsupported image format.")
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Input error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if no_interactive:
        if not food or not grams:
            console.print("[red]--no-interactive requires --food and --grams.[/red]")
            raise typer.Exit(code=1)
        label = food.strip()
        if not label:
            console.print("[red]--food must be non-empty.[/red]")
            raise typer.Exit(code=1)
        try:
            grams_value = parse_grams(grams)
        except ValueError as exc:
            console.print(f"[red]Invalid --grams:[/red] {exc}")
            raise typer.Exit(code=1) from exc
    else:
        mime_type = _get_mime_type(image_file)
        image_hash = sha256_file(image_file)

        cached = get_vision_cache(image_hash)
        if cached is None:
            with image_file.open("rb") as handle:
                image_bytes = handle.read()
            with console.status("[bold green]Analyzing image...[/bold green]"):
                try:
                    raw = analyze_image_json(image_bytes=image_bytes, mime_type=mime_type)
                except Exception as exc:  # pragma: no cover - external API
                    console.print(f"[red]Vision API error:[/red] {exc}")
                    raise typer.Exit(code=2) from exc
            set_vision_cache(image_hash, raw)
        else:
            raw = cached

        try:
            vision = normalize_candidates(raw)
        except ValueError as exc:
            console.print(f"[red]Vision response error:[/red] {exc}")
            raise typer.Exit(code=2) from exc

        label = _prompt_label(vision.items)
        grams_value = _prompt_grams()

    with console.status("[bold green]Looking up nutrition...[/bold green]"):
        try:
            match = lookup_usda_food(label, data_dir=data_dir)
        except Exception as exc:
            console.print(f"[red]Nutrition lookup error:[/red] {exc}")
            raise typer.Exit(code=2) from exc

    console.print(
        Panel.fit(
            f"[bold]{match.description}[/bold]\nUSDA FDC ID: {match.fdc_id}",
            title="Matched database item",
            border_style="blue",
        )
    )

    _render_macros(grams_value, match)

    if out is not None:
        console.print("[dim]Note:[/dim] Label rendering is implemented in Phase 7.")


if __name__ == "__main__":
    app()
