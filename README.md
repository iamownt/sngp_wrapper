## Implementation of [A Simple Approach to Improve Single-Model Deep Uncertainty via Distance-Awareness](https://arxiv.org/abs/2205.00403)

**TL;DL:** A simple approach to improve single-model deep uncertainty by replacing the last layer with a Gaussian process layer. Spectral normalization is applied to any model. Online demo is available at [Google Colab](https://colab.research.google.com/drive/1xhGez-dCh4Ggw_ikGEGD7rfia4cpzjow?usp=sharing)

## Install

It's a preview version, so you can install it from the source code.
```
pip install -e .
```

## Example
```
# Step 1: import the module

from sngp_wrapper.covert_utils import convert_to_sn_my, replace_layer_with_gaussian

# define args, like args.spec_norm_bound, args.spec_norm_replace_list, args.gaussian_process


GP_KWARGS = {
    'num_inducing': 1024,
    'gp_scale': 1.0,
    'gp_bias': 0.,
    'gp_kernel_type': 'gaussian',
    'gp_input_normalization': True,
    'gp_cov_discount_factor': -1,
    'gp_cov_ridge_penalty': 1.,
    'gp_output_bias_trainable': False,
    'gp_scale_random_features': False,
    'gp_use_custom_random_features': True,
    'gp_random_feature_type': 'orf',
    'gp_output_imagenet_initializer': True,
}
model = ...
+ if args.spec_norm_bound is not None:
+    model = convert_to_sn_my(model, args.spec_norm_replace_list, args.spec_norm_bound)
+ if args.gaussian_process:
+    replace_layer_with_gaussian(container=model, signature="classifier", **GP_KWARGS)
    
# Step 2: train the model
train_loader, val_loader = ..., ...
optimizer = ....
criterion = ...
for epoch in range(args.epochs):
    model.train()
+    if args.gaussian_process:
+        # GP_KWARGS["gp_cov_discount_factor"] == -1, in fact, it is not necessary when momentum != -1
+        model.classifier.reset_covariance_matrix()
+        kwargs = {'return_random_features': False, 'return_covariance': False,
+                  'update_precision_matrix': True, 'update_covariance_matrix': False}
+    else:
+        kwargs = {}
    for idx, assets in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        preds = model(batch["input"], **kwargs)
        loss = criterion(preds, batch["target"])
        loss.backward()
        optimizer.step()
 
# Step 3: evaluate or test the model
    model.eval()
    logit_list, uncertainty_list, label_list = [], [], []
+    if args.gaussian_process:
+        model.classifier.update_covariance_matrix()
+        eval_kwargs = {'return_random_features': False, 'return_covariance': True,
+                       'update_precision_matrix': False, 'update_covariance_matrix': False}
+   else:
+        eval_kwargs = {}
    for idx, batch in tqdm(enumerate(val_loader)):
        output = model(batch["input"], **eval_kwargs)
        if isinstance(batch["input"], tuple):
            logits, covariance = output
            logits = logits.cpu().detach().numpy()
            uncertainty = torch.diagonal(covariance).cpu().detach().numpy()
            uncertainty_list.extend(uncertainty)
        else:
            logits = output.cpu().detach().numpy()
        logit_list.extend(logits)
        label_list.extend(assets['labels'].long())
    # save or evalute the model with logit_list, uncertainty_list, label_list
```

## Acknowledgements
We would like to thank the authors of the original paper for their work and the authors of the [uncertainty-baselines](https://github.com/google/uncertainty-baselines), and [ml-sigma-reparam](https://github.com/apple/ml-sigma-reparam).
