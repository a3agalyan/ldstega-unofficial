## TLDR:
My aim was to replicate the algorithm described in paper, but could not achieve similar results. Same level of accuracy was not achieved at any parameters sweep and also algorithm showed heavy instablity across different seeds over 10% std of accuracy for 1024 bits length.

PS: Authors do not state exact model they used, so exact replication is not possible, so I did not bother myself using mentioned dataset for benchmarks, as it seemed irrelevant. By the lantent space mentioned equal to 32 × 32 × 4 and image size of 256x256, one can assume that model was LDM `CompVis/ldm-text2im-large-256`.  

## What authors did:
They used LDM to embed secret bits into diffusion-generated images by manipulating latent values at the final DDIM step Z_T. To address the robustness of steganography, a
parameter with a truncated interval $\gamma$ is to ensure robust extraction (`bit accuracy`) and visual quality (`PNSR`).

## What authors discovered:
The main idea behind paper is that doing VAE-roundtrip: 
$$
Z_T \rightarrow \text{encode to pixels} \rightarrow \text{PNG/JPEG compression} \rightarrow \text{decode to latent space} \rightarrow Z'_T
$$
then calculate discrepancy score $D = |Z_T − Z'_T|$ and thus you can calculate stable positions favourable for decoding. The major idea is the by embedding secret bits into most stable coordinates we can minimize payload data loss caused by the reconstruction of the latent space and the lossy transmission of stego image.

Message hiding is controled by truncation parameter $\gamma$ that defines sampling interval of truncated Gaussian distributions guided by secret data
(−∞, 𝜇𝑇 −1 −𝛾) and (𝜇𝑇 −1 + 𝛾, +∞). Then driven by the secret data, one candidate pool is selected as
the sampling interval.

LDStega leverages the Gaussian distribution property of $Z_T$ to hide the encrypted data. The idea is to mimic original Gaussian distribution with subtle truncated distribution. For $\gamma$ <= 0.3 paper stats to be optimum of robustness with no visual degradation. 
![alt text](images/trunc_distribution.png)

## What I discovered:

Authors do not state exact model they used, however one can assume all the LDM family models behave similarly in this setup. However I discovered that the accuracy heavily corellates with latent size.
| Model | HuggingFace ID | Latent Space Size |
|-------|---------------|--------|
| Stable Diffusion 1.5 | `runwayml/stable-diffusion-v1-5` | Tested |
| Stable Diffusion 2.1 | `stabilityai/stable-diffusion-2-1` | Tested |
| SDXL | `stabilityai/stable-diffusion-xl-base-1.0` | Tested |
| LDM | `CompVis/ldm-text2im-large-256` | Tested |
---


### Stable coordinates state
Observed values do match with once stated in paper. However it seems like they do not remain same across different round-trips.


| My experiment | Paper reference |
|-------------|-------------|
| <img src="images/d_barplot.png" width="400" height="150"> | <img src="images/paper_reference.png" width="400" height="150"> |

Across 4 models that I tested 4 Model were tested by me. 

In my experiments I achieved similar optimum as in paper for gamma at 0.3.

<img src="images/gamma_optimum.png" width="400" height="150"> 

SDXL showed most promising results but even they are far from stated in paper.

<img src="images/sdxl_res.png" width="400" height="150"> 
### 