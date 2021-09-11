{
    BiMeanVAE(latent_dim, free_bit)::
        local embedding_dim = 256;
        local hidden_size = 512;
        local num_layers = 1;
        {
            "type": "bimeanvae",
            "embedding_dim": embedding_dim,
            "hidden_size": hidden_size,
            "latent_dim": latent_dim,
            "num_layers": num_layers,
            "free_bit": free_bit
        },

    Optimus(latent_dim, free_bit)::
        {
            "type": "optimus",
            "latent_dim": latent_dim,
            "free_bit": free_bit,
        },

    VAETrainer(num_steps, checkout_step, batch_size, lr)::
        {
            "num_steps": num_steps,
            "checkout_step": checkout_step,
            "batch_size": batch_size,
            "lr": lr
        }
}