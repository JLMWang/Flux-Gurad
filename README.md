# Flux-Gurad
Flux-Guard: Flow-Guided Adversarial Face Editing for Identity Privacy Protection
Abstract
The widespread deployment of face recognition (FR) systems exposes personal images shared on social media and public platforms to identity linkage and privacy risks. Existing adversarial privacy protection methods can degrade unauthorized FR performance but are not compatible with generative face editing. Artificial intelligence-driven face editing tools are gaining popularity, which has significantly increased user demand for personalized portrait generation and social sharing. However, current editing methods often preserve identity features, making the edited images still susceptible to tracking by malicious FR systems. Thus, this paper proposes Flux-Guard, a privacy-preserving face editing framework based on adversarial attacks, which integrates face editing and privacy protection within a unified generative process. Specifically, we design a flow trajectory control method to align semantic manipulations with the generative process and introduce latent-space adversarial optimization with an adaptive perceptual-loss-driven weighting strategy, dynamically adjusting adversarial strength to maximize attack effectiveness while preserving visual quality.

Extensive experiments demonstrate that Flux-Guard supports face editing while significantly improving attack success rates against cross-domain face recognition models on the CelebA-HQ and LADN datasets. Furthermore, evaluation results for commercial APIs have confirmed its effectiveness in real-world applications. 

Download the weights for victim models from [here](https://drive.google.com/file/d/19_Y0jR789BGciogjjoGtWNEv-5QBiCB7/view) and extract them to `./models`.
## Run the code

```bash
python runfluxguard.py
```
