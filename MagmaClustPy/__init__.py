import logging

from MagmaClustPy.backend import DefaultNumPyBackend, MLXBackend


class LinAlgBackend:
	def __init__(self):
		self.current_backend = DefaultNumPyBackend()

	def change_backend(self, backend: str):
		match backend.lower():
			case "numpy":
				self.current_backend = DefaultNumPyBackend()
			case "mlx":
				self.current_backend = MLXBackend()
			case _:
				raise ValueError(f"Backend {backend} not supported.")
		logging.info(f"Changed backend to {backend}.")

	def __getattr__(self, item):
		return getattr(self.current_backend, item)


lin_alg_backend = LinAlgBackend()

config = {
	"pen_diag": 1e-10,
}
