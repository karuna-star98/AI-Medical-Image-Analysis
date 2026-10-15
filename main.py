"""Command-line entry point for the AI-Powered Medical Image Analysis project.

Typical flow (see README):
  python main.py synthetic                      # optional: make test data (no download)
  python main.py explore                        # dataset stats + sample images
  python main.py preprocess                     # preprocessing demo figure
  python main.py train --model cnn --epochs 15
  python main.py evaluate --model-path models/cnn_best.keras
  python main.py predict  --model-path models/cnn_best.keras --image some_xray.jpeg --gradcam
  python main.py triage   --model-path models/cnn_best.keras --folder data/chest_xray/test
  python main.py serve    --model-path models/cnn_best.keras
"""
import argparse
import os
from pathlib import Path

from src.config import CLASS_NAMES, DISCLAIMER, MODELS_DIR, OUTPUTS_DIR, REAL_DIR, SIM_DIR, ensure_dirs


def data_dir_from(args):
    return SIM_DIR if args.dataset == "simulated" else REAL_DIR


def cmd_synthetic(args):
    from src.synthetic import generate_dataset
    print("Generating synthetic dataset (cartoon images - pipeline testing only)...")
    print(generate_dataset(SIM_DIR, args.n_train, args.n_val, args.n_test))
    print(f"Done -> {SIM_DIR}. Use `--dataset simulated` with other commands.")


def cmd_explore(args):
    from src.data_loader import dataset_summary
    from src.visualize import plot_class_distribution, show_samples
    ensure_dirs()
    d = data_dir_from(args)
    summary = dataset_summary(d)
    for split, counts in summary.items():
        print(f"{split:>6}: {counts}")
    print(plot_class_distribution(summary, OUTPUTS_DIR / "class_distribution.png"))
    print(show_samples(d, CLASS_NAMES, OUTPUTS_DIR / "sample_images.png"))


def cmd_preprocess(args):
    from src.preprocessing import list_images, preprocessing_demo
    ensure_dirs()
    image = Path(args.image) if args.image else list_images(data_dir_from(args) / "train" / "PNEUMONIA")[0]
    print(preprocessing_demo(image, OUTPUTS_DIR / "preprocessing_steps.png"))


def cmd_train(args):
    from src.train import train
    train(data_dir_from(args), args.model, args.epochs, args.fine_tune_epochs,
          use_class_weights=not args.no_class_weights, tag=args.tag)


def cmd_evaluate(args):
    from src.evaluate import evaluate
    evaluate(Path(args.model_path), data_dir_from(args))


def cmd_predict(args):
    from src.predict import load_trained_model, predict_image
    model = load_trained_model(args.model_path)
    result = predict_image(model, args.image)
    print(result); print(DISCLAIMER)
    if args.gradcam:
        from src.explain import save_gradcam_figure
        ensure_dirs()
        out = OUTPUTS_DIR / f"gradcam_{Path(args.image).stem}.png"
        print("Grad-CAM saved:", save_gradcam_figure(model, args.image, out, result))


def cmd_triage(args):
    from src.triage import run_triage
    run_triage(Path(args.model_path), args.folder)


def cmd_serve(args):
    os.environ["MODEL_PATH"] = str(args.model_path)
    from src.app import app
    print(DISCLAIMER)
    app.run(host=args.host, port=args.port, debug=False)


def build_parser():
    p = argparse.ArgumentParser(description="AI-Powered Medical Image Analysis")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--dataset", choices=["real", "simulated"], default="real",
                        help="real = data/chest_xray, simulated = data/simulated")
        sp.set_defaults(func=fn)
        return sp

    sp = add("synthetic", cmd_synthetic, "generate synthetic test data")
    sp.add_argument("--n-train", type=int, default=400)
    sp.add_argument("--n-val", type=int, default=80)
    sp.add_argument("--n-test", type=int, default=120)

    add("explore", cmd_explore, "dataset statistics + sample images")

    sp = add("preprocess", cmd_preprocess, "preprocessing demo figure")
    sp.add_argument("--image", help="optional path to one image")

    sp = add("train", cmd_train, "train a model")
    sp.add_argument("--model", choices=["cnn", "mobilenet"], default="cnn")
    sp.add_argument("--epochs", type=int, default=15)
    sp.add_argument("--fine-tune-epochs", type=int, default=0, help="mobilenet only")
    sp.add_argument("--no-class-weights", action="store_true")
    sp.add_argument("--tag", help="name used for saved files (default = model type)")

    sp = add("evaluate", cmd_evaluate, "evaluate on the test set")
    sp.add_argument("--model-path", default=str(MODELS_DIR / "cnn_best.keras"))

    sp = add("predict", cmd_predict, "predict a single image")
    sp.add_argument("--model-path", default=str(MODELS_DIR / "cnn_best.keras"))
    sp.add_argument("--image", required=True)
    sp.add_argument("--gradcam", action="store_true", help="save a Grad-CAM heatmap (cnn only)")

    sp = add("triage", cmd_triage, "virtual-hospital worklist simulation")
    sp.add_argument("--model-path", default=str(MODELS_DIR / "cnn_best.keras"))
    sp.add_argument("--folder", required=True)

    sp = add("serve", cmd_serve, "start the Flask web/API demo")
    sp.add_argument("--model-path", default=str(MODELS_DIR / "cnn_best.keras"))
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=5000)
    return p


```python
if __name__ == "__main__":
    ensure_dirs()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "serve":
        args.host = "0.0.0.0"
        args.port = int(os.environ.get("PORT", args.port))

    args.func(args)
