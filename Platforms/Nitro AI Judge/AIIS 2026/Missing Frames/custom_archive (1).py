

######################################

# Read and process the training data
train_data = pd.read_csv('train_data.csv')
train_data['answer'] = train_data['answer'].apply(lambda x: np.array(ast.literal_eval(x)).astype(np.float32))

# Before sumbitting apply the follosing processing on the column of arrays
submission_csv['answer'] = submission_csv['answer'].apply(lambda x: x.tolist())

######################################

# Run to ensure consistency

# Autoencoder model
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
vae.to(device, dtype=torch.float32)
vae.eval()
vae.enable_tiling()
vae.enable_slicing()


######################################


# Decode a latent
with torch.no_grad():
  img = vae.decode(latents).sample[0]

# Encode an image
with torch.no_grad():
  latent = vae.encode(timg)['latent_dist'].sample()


######################################

# Depth model
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf", device=device)
# Get the depth of an image
depth = pipe(image)['depth'].convert('RGB') # image - PIL.Image object


######################################
