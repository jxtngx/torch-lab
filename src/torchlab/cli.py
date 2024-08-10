# Copyright Justin R. Goheen.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path

import pandas as pd
import typer
from rich.pretty import pprint
from typing_extensions import Annotated

from torchlab.utils.config import load_config

cfg = load_config(os.path.join(os.getcwd(), "training-config.yaml"))

app = typer.Typer()
# DOCS
docs_app = typer.Typer()
app.add_typer(docs_app, name="docs")
# DATA
data_app = typer.Typer()
app.add_typer(data_app, name="data")
# TRAINER
trainer_app = typer.Typer()
app.add_typer(trainer_app, name="trainer")
# TUNER
tune_app = typer.Typer()
app.add_typer(tune_app, name="hpo")


@app.callback()
def callback() -> None:
    pass


# #### TRAINER ##### #
@trainer_app.command("fit")
def trainer_fit_command() -> None:
    pass


# #### DATA #### #
@data_app.command("run")
def data_run_command() -> None:
    pass


# #### DOCS #### #
@docs_app.command("build")
def build_docs() -> None:
    import shutil

    os.system("mkdocs build")
    shutil.copyfile(src="README.md", dst="docs/index.md")


@docs_app.command("serve")
def serve_docs() -> None:
    os.system("mkdocs serve")
