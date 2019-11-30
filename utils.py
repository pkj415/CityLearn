class Discretizer():
	def __init__(self, min_val, max_val, level_cnt):
		self.min_val = min_val
		self.max_val = max_val
		self.level_cnt = level_cnt
		vals = []
		for i in range(level_cnt):
			vals.append(self.get_val(i))
		print("Disc values {0}".format(vals))


	# Gives level just below val (flooring)
	def get_level(self, val):
		slab_size = (self.max_val - self.min_val)/(self.level_cnt-1)
		return int((val - self.min_val)/slab_size)

	# Gives val of level
	def get_val(self, level):
		slab_size = (self.max_val - self.min_val)/(self.level_cnt-1)
		return slab_size*level + self.min_val