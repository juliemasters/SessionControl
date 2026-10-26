# Official WISE source

These modules are copied **without modification** from [zjunlp/EasyEdit](https://github.com/zjunlp/EasyEdit), revision `4c109870955a4522ac3d7cf10ad00f34de8e4f0d`.

`UPSTREAM.json` records the original `easyeditor/` source root and each copied file's SHA-256. `LICENSE` is the upstream MIT license. The modules retain their original relative imports. A lightweight integration loader imports just WISE and its dependencies, bypassing EasyEdit's unrelated optional model editors.

Dataset adaptation, masked-loss projection, frozen-prefix execution caching and tensor-only checkpointing live outside these files in [`experiments/wise_qa500`](../../experiments/wise_qa500). The upstream WISE optimizer, activation loss, masks, native router and sharding functions are unchanged.
