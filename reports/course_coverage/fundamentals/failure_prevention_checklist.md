# Controlled Failure Prevention Checklist

These cases use train-only copies and validation-only evaluation. No locked test data is accessed.

## unscaled_images

- Observed: Validation Macro F1=0.1667 versus the correct-loop reference 0.2886; train loss=1.0987544059753418.
- Prevention: Keep explicit image scaling in the model and assert input ranges.

## excessive_learning_rate

- Observed: Validation Macro F1=0.1667 versus the correct-loop reference 0.2886; train loss=1.13113272190094.
- Prevention: Use bounded predefined learning rates and terminate on NaN.

## tiny_learning_rate

- Observed: Validation Macro F1=0.2558 versus the correct-loop reference 0.2886; train loss=1.1205341815948486.
- Prevention: Track loss reduction and reject configurations with negligible progress.

## excessive_dropout

- Observed: Validation Macro F1=0.1667 versus the correct-loop reference 0.2886; train loss=1.0986475944519043.
- Prevention: Treat dropout as a tuned regularizer rather than a default maximum.

## misaligned_train_labels

- Observed: Validation Macro F1=0.2668 versus the correct-loop reference 0.2886; train loss=1.101723551750183.
- Prevention: Verify sample IDs and labels before every shuffle or batching operation.

## sigmoid_activation

- Observed: Validation Macro F1=0.2402 versus the correct-loop reference 0.2886; train loss=1.1439261436462402.
- Prevention: Compare timing and gradients; prefer ReLU-like activations for hidden layers.

## deep_sigmoid_gradient_probe

- Observed: Deep sigmoid networks can produce strongly uneven gradient norms; the measured first/last kernel ratio is reported without forcing a result.
- Prevention: Prefer ReLU-like activations, normalization, and residual paths for deep stacks.

## missing_optimizer_step_probe

- Observed: Repeated forward passes without an optimizer update left all weights unchanged and did not create systematic learning.
- Prevention: Use model.fit/train_on_batch or a reviewed custom loop that applies gradients exactly once per batch. Keras manages gradient reset inside its built-in training step.

## validation_training_blocked

- Observed: The suite exposes fixed train and validation arrays; every fit call uses train inputs for updates and validation inputs only through validation_data.
- Prevention: Keep the split paths fixed in configuration and audit that validation evaluation leaves weights unchanged.
