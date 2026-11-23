# References and Further Reading

All peer-reviewed papers, technical articles, compiler specifications, and open-source weight-space frameworks cited across Chapters 1 through 8 are cataloged below in numerical order (`[1]`–`[40]`).

## Transformer Foundations, Low-Rank Adapters, and Quantization

- **[1]** Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, Illia (2017). "Attention Is All You Need." *Advances in Neural Information Processing Systems (NeurIPS)*, Vol. 30 (`arXiv:1706.03762`).
- **[2]** Shazeer, N. (2020). "GLU Variants Improve Transformer." Google Research Technical Report (`arXiv:2002.05202`).
- **[3]** Zhang, B., and Sennrich, R. (2019). "Root Mean Square Layer Normalization." *Advances in Neural Information Processing Systems (NeurIPS)*, Vol. 32 (`arXiv:1910.07467`).
- **[4]** Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. (2024). "RoFormer: Enhanced Transformer with Rotary Position Embedding." *Neurocomputing*, Vol. 568, Article 127063 (`doi:10.1016/j.neucom.2023.127063`; `arXiv:2104.09864`).
- **[5]** Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., et al. (2023). "LLaMA: Open and Efficient Foundation Language Models." Meta AI Technical Report (`arXiv:2302.13971`).
- **[6]** Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, Ilya (2019). "Language Models are Unsupervised Multitask Learners." OpenAI Technical Report (`GPT-2` architecture and `Conv1D` parameterization).
- **[7]** Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." *International Conference on Learning Representations (ICLR)* (`arXiv:2106.09685`).
- **[8]** Kalajdzievski, D. (2023). "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA." (`rsLoRA`, `arXiv:2312.03732`).
- **[9]** Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." *Advances in Neural Information Processing Systems (NeurIPS)*, Vol. 36 (`NF4` quantization, `arXiv:2305.14314`).
- **[10]** Mangrulkar, S., Gugger, S., Debut, L., Belkada, Y., Paul, S., and Bossan, B. (2022). *PEFT: State-of-the-Art Parameter-Efficient Fine-Tuning Methods*. Hugging Face open-source library (`github.com/huggingface/peft`).

## Paradigm 1: Constructive Compilation, Attention Sinks, and Compiler IRs

- **[11]** Weiss, G., Goldberg, Y., and Yahav, E. (2021). "Thinking Like Transformers." *Proceedings of the 38th International Conference on Machine Learning (ICML)* (`RASP`, `arXiv:2106.06981`).
- **[12]** Lindner, D., Kramár, J., Farquhar, S., Rahtz, M., McGrath, T., and Mikulik, V. (2023). "Tracr: Compiled Transformers as a Laboratory for Interpretability." *Advances in Neural Information Processing Systems (NeurIPS)*, Vol. 36 (`arXiv:2301.05062`; `github.com/google-deepmind/tracr`).
- **[13]** Shaw, P., Cohan, J., Eisenstein, J., Toutanova, K., and Weiss, G. (2024). "ALTA: Compiler-Based Analysis of Transformers." *Transactions on Machine Learning Research (TMLR)* (`arXiv:2410.18077`; `github.com/google-deepmind/alta`).
- **[14]** Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. (2024). "Efficient Streaming Language Models with Attention Sinks." *International Conference on Learning Representations (ICLR)* (`StreamingLLM`, `arXiv:2309.17453`).
- **[15]** Reissmann, N., Meyer, J. C., Bahmann, H., and Själander, M. (2020). "RVSDG: An Intermediate Representation for Optimizing Compilers." *ACM Transactions on Architecture and Code Optimization (TACO)*, Vol. 17, No. 4, Article 28 (`doi:10.1145/3419088`).
- **[16]** Lattner, C., Amini, M., Bondhugula, U., Cohen, A., Davis, A., Pienaar, J., et al. (2021). "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation." *IEEE/ACM International Symposium on Code Generation and Optimization (CGO)*, pp. 2–14 (`arXiv:2002.11054`).

## Paradigm 4 (Part A): Mechanistic Circuits, Effective Rank, and Cold Auditing

- **[17]** Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., et al. (2021). "A Mathematical Framework for Transformer Circuits." *Transformer Circuits Thread* (`transformer-circuits.pub/2021/framework`).
- **[18]** Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., et al. (2022). "In-context Learning and Induction Heads." *Transformer Circuits Thread* (`arXiv:2209.11895`).
- **[19]** Nanda, N., and Bloom, J. (2022). *TransformerLens: A Library for Mechanistic Interpretability of Generative Language Models*. Open-source library (`github.com/TransformerLensOrg/TransformerLens`).
- **[20]** Fiotto-Kaufman, J., Loftus, A. R., Todd, E., Brinkmann, J., Juang, C., Pal, K., et al. (2024). "NNsight and NDIF: Democratizing Access to Foundation Model Internals." (`arXiv:2407.14561`; `github.com/ndif-team/nnsight`).
- **[21]** Roy, O., and Vetterli, M. (2007). "The Effective Rank: A Measure of Effective Dimensionality." *Proceedings of the 15th European Signal Processing Conference (EUSIPCO)*, pp. 606–610.
- **[22]** Hugging Face (2023). *Safetensors: Simple, Safe Way to Store and Distribute Tensors*. Wire specification and open-source container library (`github.com/huggingface/safetensors`).

## Paradigm 2a: Associative Memory and Null-Space Knowledge Editing

- **[23]** Meng, K., Bau, D., Andonian, A., and Belinkov, Y. (2022). "Locating and Editing Factual Associations in GPT." *Advances in Neural Information Processing Systems (NeurIPS)*, Vol. 35 (`ROME`, `arXiv:2202.05262`).
- **[24]** Meng, K., Sharma, A. S., Andonian, A., Belinkov, Y., and Bau, D. (2023). "Mass-Editing Memory in a Transformer." *International Conference on Learning Representations (ICLR)* (`MEMIT`, `arXiv:2210.07229`).
- **[25]** Fang, J., Jiang, H., Wang, K., Ma, Y., Shi, J., Wang, X., He, X., and Chua, T.-S. (2025). "AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models." *International Conference on Learning Representations (ICLR)* (`arXiv:2410.02355`; `github.com/jianghoucheng/AlphaEdit`).
- **[26]** Wang, P., Zhang, N., Xie, X., Yao, Y., Tian, B., Wang, M., et al. (2024). "EasyEdit: An Easy-to-use Knowledge Editing Framework for Large Language Models." *Proceedings of ACL 2024 (System Demonstrations)* (`arXiv:2308.07269`; `github.com/zjunlp/EasyEdit`).

## Paradigm 2b: Directional Subspace Surgery and Rank Truncation

- **[27]** Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., and Nanda, N. (2024). "Refusal in Language Models Is Mediated by a Single Direction." *Advances in Neural Information Processing Systems (NeurIPS)*, Vol. 37 (`arXiv:2406.11717`).
- **[28]** Labonne, M. (2024). "Uncensor any LLM with abliteration." Hugging Face Technical Blog, June 2024 (`huggingface.co/blog/mlabonne/abliteration`).
- **[29]** Sharma, P., Ash, J. T., and Misra, D. (2024). "The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction." *International Conference on Learning Representations (ICLR)* (`LASER`, `arXiv:2312.13558`; `github.com/pratyushasharma/laser`).

## Paradigm 2c & Paradigm 3: Weight Algebra, Re-Basin, and Meta-Operators

- **[30]** Ilharco, G., Ribeiro, M. T., Wortsman, M., Gururangan, S., Schmidt, L., Hajishirzi, H., and Farhadi, A. (2023). "Editing Models with Task Arithmetic." *International Conference on Learning Representations (ICLR)* (`arXiv:2212.04089`).
- **[31]** Yadav, P., Tam, D., Choshen, L., Raffel, C., and Bansal, M. (2023). "TIES-Merging: Resolving Interference When Merging Models." *Advances in Neural Information Processing Systems (NeurIPS)*, Vol. 36 (`arXiv:2306.01708`).
- **[32]** Yu, L., Yu, B., Yu, H., Huang, F., and Li, Y. (2024). "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch." *Proceedings of the 41st International Conference on Machine Learning (ICML)* (`DARE`, `arXiv:2311.03099`).
- **[33]** Goddard, C., Siriwardhana, S., Ehghaghi, M., Meyers, L., Karpukhin, V., Benedict, B., McQuade, M., and Solawetz, J. (2024). "Arcee's MergeKit: A Toolkit for Merging Large Language Models." (`arXiv:2403.13257`; `github.com/arcee-ai/mergekit`).
- **[34]** Ainsworth, S. K., Hayase, J., and Srinivasa, S. (2023). "Git Re-Basin: Merging Models modulo Permutation Symmetries." *International Conference on Learning Representations (ICLR)* (`arXiv:2209.04836`; `github.com/samuela/git-re-basin`).
- **[35]** Charakorn, R., Cetin, E., Tang, Y., and Lange, R. T. (2025). "Text-to-LoRA: Instant Transformer Adaption." *Proceedings of ICML* (`arXiv:2506.06105`; `github.com/SakanaAI/text-to-lora`); and *Doc-to-LoRA: Learning to Instantly Internalize Contexts* (`arXiv:2602.15902`).
- **[36]** Navon, A., Shamsian, A., Achituve, Idan, Fetaya, E., Chechik, G., and Maron, H. (2023). "Equivariant Architectures for Learning in Deep Weight Spaces." *Proceedings of ICML* (`DWSNets`, `arXiv:2301.12780`); and Zhou, A., et al. (2023), "Permutation Equivariant Neural Functionals," *NeurIPS* (`NFNs`, `arXiv:2302.14040`).

## Paradigm 4 (Part B): Constrained Decoding, SMT Verification, and Integrity

- **[37]** Willard, B. T., and Louf, R. (2023). "Efficient Guided Generation for Large Language Models." (`Outlines`, `arXiv:2307.09702`); and Ugare, S., et al. (2024), "SynCode: LLM Generation with Grammar-Augmented Lookahead," *ICSE 2024* (`arXiv:2403.01632`).
- **[38]** de Moura, L., and Bjørner, N. (2008). "Z3: An Efficient SMT Solver." *Tools and Algorithms for the Construction and Analysis of Systems (TACAS)*, LNCS Vol. 4963, pp. 337–340 (`doi:10.1007/978-3-540-78800-3_24`).
- **[39]** Necula, G. C. (1997). "Proof-Carrying Code." *Proceedings of ACM POPL*, pp. 106–119 (`doi:10.1145/263699.263712`); and Appel, A. W. (2001), "Foundational Proof-Carrying Code," *IEEE LICS*, pp. 247–256 (`doi:10.1109/LICS.2001.932500`).
- **[40]** Willsey, M., Nandi, C., Wang, Y. R., Flatt, O., Pailoor, S., and Tatlock, Z. (2021). "egg: Fast and Extensible Equality Saturation." *PACMPL (POPL)*, Vol. 5, Article 23 (`arXiv:2004.03082`); and O'Connor, J., Aumasson, J.-P., Neves, S., and Wilcox-O'Hearn, Z. (2020), "BLAKE3: One Function, Fast Everywhere," *IACR ePrint* `2020/864`.

---

*From **Weight-Space Engineering** (1st Edition, ISBN `979-8-1447-3586-5`) by Aguentil H. — [Read Chapter 1 Free](https://weight-space-engineering.weight-space-engineering.workers.dev/chapter-1/) | [Get the Complete Handbook + 8-Lab Python Suite](https://whop.com/weight-space-engineering/weight-space-engineering-handbook-python-lab-suite/)*
