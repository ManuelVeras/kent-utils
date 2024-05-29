import tensorflow as tf
import tensorflow_probability as tfp

# Set up the distributions
d = 3  # Dimensionality
mu_p = tf.constant([1.0, 0.0, 0.0], dtype=tf.float32)  # Mean direction of p
kappa_p = tf.constant(2.0, dtype=tf.float32)  # Concentration parameter of p
mu_q = tf.constant([0.0, 1.0, 0.0], dtype=tf.float32)  # Mean direction of q
kappa_q = tf.constant(3.0, dtype=tf.float32)  # Concentration parameter of q

# Define the von Mises-Fisher distributions
vmf_p = tfp.distributions.VonMisesFisher(mean_direction=mu_p, concentration=kappa_p)
vmf_q = tfp.distributions.VonMisesFisher(mean_direction=mu_q, concentration=kappa_q)

# Calculate the KL divergence
kl_divergence = tfp.distributions.kl_divergence(vmf_p, vmf_q)

print(f"KL divergence: {kl_divergence.numpy()}")
