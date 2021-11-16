import time
import logging
import sys

def main():
	logger = logging.getLogger()
	logger.handlers.clear()
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
	stdout_handler = logging.StreamHandler(sys.stdout)
	stdout_handler.setLevel(logging.DEBUG)
	stdout_handler.setFormatter(formatter)

	file_handler = logging.FileHandler("test.log")
	file_handler.setLevel(logging.DEBUG)
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	logger.addHandler(stdout_handler)

	for i in range(10):
		logger.info(i)

	with open("test.log") as file:
		for line in file:
			print(line)


if __name__ == "__main__":
	main()