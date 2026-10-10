"""NB-07: minimal training loop.

VERIFICATION STATUS: not executed -- see ``training/__init__.py``.

Deliberately a single small class, not a framework: no callback
system, no experiment-tracking integration, no distributed strategy
abstraction (per the parent milestone's explicit "no sophisticated
experiment tracking", "no distributed training" scope limits). It
supports exactly the sequence the milestone's "First Training Proof"
section asks for: forward -> loss -> backward -> optimizer step ->
zero_grad -> metrics, plus periodic evaluation and checkpointing.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator

import torch

from training.checkpoint import save_checkpoint
from training.config import TrainingConfig
from training.dataset import iter_lazy_batches
from training.errors import TrainingError
from training.loss import causal_lm_loss, evaluation_nll_count


class Trainer:
    """Runs a deterministic training loop over repeatable ordered sequences.

    Not responsible for constructing the model, optimizer, or
    dataset -- all three are supplied already-built, matching the
    same "this module is an integration boundary, not another
    implementation" principle used throughout this milestone sequence
    (see ``training/checkpoint.py``'s docstring for the same point
    made about model construction).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        model_config,  # yug.native.config.ModelConfig
        tokenizer_identity: str,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.model_config = model_config
        self.tokenizer_identity = tokenizer_identity
        self.step = 0
        # The caller constructs and owns the model (see class docstring);
        # this is the one exception -- moving it to the configured
        # device here, once, at construction, is what makes
        # config.device meaningful at all. Without this, config.device
        # was a field nothing ever read (found during NB-07 Phase 2
        # review: batches built by training.dataset are always plain
        # CPU tensors, and nothing moved them or the model anywhere).
        self.model.to(self.config.device)

    def train_step(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> float:
        """Run one forward/loss/backward/optimizer-step/zero_grad
        cycle on a single batch. Returns the scalar loss value (a
        plain Python ``float``, already detached from the graph).

        ``input_ids``/``target_ids`` are moved to ``self.config.device``
        here -- ``training.dataset`` deliberately never sets a device
        on the tensors it builds (the standard dataset/DataLoader-vs-
        training-loop split: dataset produces host tensors, the loop
        moves them to the execution device), so this is the one place
        that transfer needs to happen.
        """
        input_ids = input_ids.to(self.config.device)
        target_ids = target_ids.to(self.config.device)

        self.model.train()
        logits = self.model(input_ids)
        loss = causal_lm_loss(logits, target_ids)

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm
            )
        self.optimizer.step()

        self.step += 1
        return loss.item()

    def evaluate(self, batches: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Global valid-target mean loss with no gradient computation.
        Restores the model's prior ``training`` mode afterward, the
        same convention ``yug.native.generation.generate`` already
        uses for the same reason (a caller mid-training-loop should
        not have their model silently left in eval mode)."""
        was_training = self.model.training
        self.model.eval()
        total_nll = 0.0
        total_valid_targets = 0
        try:
            with torch.no_grad():
                for input_ids, target_ids in batches:
                    input_ids = input_ids.to(self.config.device)
                    target_ids = target_ids.to(self.config.device)
                    logits = self.model(input_ids)
                    nll, count = evaluation_nll_count(logits, target_ids)
                    total_nll += nll
                    total_valid_targets += count
        finally:
            self.model.train(was_training)
        if not total_valid_targets:
            raise TrainingError("Evaluation has zero valid targets")
        if not math.isfinite(total_nll):
            raise TrainingError("Non-finite total evaluation NLL")
        return total_nll / total_valid_targets

    def fit(
        self,
        train_sequences: Iterable[list[int]],
        eval_sequences: Iterable[list[int]],
        checkpoint_dir: str,
    ) -> list[float]:
        """Run up to ``config.max_steps`` training steps over
        ``train_sequences`` (repeating from the start if exhausted --
        a tiny fixed dataset run to convergence, per this milestone's
        "tiny overfit" proof, is expected to need multiple passes over
        the same handful of sequences), evaluating and checkpointing
        at their respective configured intervals.

        Returns the list of per-step training losses, in order --
        this is what a caller checks for the "loss decreases" /
        "overfit" acceptance criteria; this method does not itself
        assert anything about the trend, since judging convergence is
        the caller's job (the milestone's own "do not hard-code an
        arbitrary loss threshold without validating the numerical
        behavior first").
        """
        # A one-shot generator cannot provide a second epoch/evaluation.
        for sequences in (train_sequences, eval_sequences):
            if isinstance(sequences, Iterator):
                raise TrainingError("fit requires repeatable sequences, not a one-shot iterator")
        train_batches = iter_lazy_batches(train_sequences, self.config.batch_size)

        losses: list[float] = []
        while self.step < self.config.max_steps:
            try:
                input_ids, target_ids = next(train_batches)
            except StopIteration:
                train_batches = iter_lazy_batches(train_sequences, self.config.batch_size)
                try:
                    input_ids, target_ids = next(train_batches)
                except StopIteration as exc:
                    raise TrainingError("Training sequences are empty or not repeatable") from exc
            losses.append(self.train_step(input_ids, target_ids))

            if self.step % self.config.eval_interval == 0:
                self.evaluate(iter_lazy_batches(eval_sequences, self.config.batch_size))

            if self.step % self.config.checkpoint_interval == 0:
                save_checkpoint(
                    checkpoint_dir,
                    model=self.model,
                    optimizer=self.optimizer,
                    model_config=self.model_config,
                    training_config=self.config,
                    step=self.step,
                    tokenizer_identity=self.tokenizer_identity,
                )

        return losses
