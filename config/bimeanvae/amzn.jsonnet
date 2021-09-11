local lib = import '../utils.libsonnet';
local data_type = "amzn";
local latent_dim = 512;
local free_bit = 0.25;
local num_steps = 100000;
local checkout_step = 1000;
local batch_size = 256;
local lr = 1e-3;

{
    "data_dir": "./data/%s" % data_type,
    "spm_path": "./data/sentencepiece/%s.model" % data_type,
    "model": lib.BiMeanVAE(latent_dim, free_bit),
    "trainer": lib.VAETrainer(num_steps, checkout_step, batch_size, lr)
}
