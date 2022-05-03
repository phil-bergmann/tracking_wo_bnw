from __future__ import division
from collections import OrderedDict, Iterable
import pandas as pd
import numpy as np
import pickle

class Metrics(object):
	def __init__(self):
		self.metrics = OrderedDict()
		self.cache_dict = OrderedDict()


	def register(self, name=None, value=None, formatter=None,
				display_name=None, write_db = True, write_mail = True):
		"""Register a new metric.
		Params
		------
		name: str
			Name of the metric. Name is used for computation and set as attribute.
        display_name: str or None
            Disoplay name of variable written in db and mail
		value:
		formatter:
			Formatter to present value of metric. E.g. `'{:.2f}'.format`
		write_db: boolean, default = True
			Write value into db
		write_mail: boolean, default = True
			Write metric in result mail to user
		"""
		assert not name is None, 'No name specified'.format(name)

		if not value:
			value = 0

		self.__setattr__( name, value)

		if not display_name: display_name = name
		self.metrics[name] = {
		    'name' : name,
		    'write_db' : write_db,
		    'formatter' : formatter,
		    'write_mail' : write_mail,
		    'display_name' : display_name
		}

	def cache(self, name=None, value=None, func=None):
		assert not name is None, 'No name specified'.format(name)


		self.__setattr__( name, value)


		self.cache_dict[name] = {
		    'name' : name,
		    'func' : func
		}



	def __call__(self, name):
		return self.metrics[name]

	@property
	def names(self):
		"""Returns the name identifiers of all registered metrics."""
		return [v['name'] for v in self.metrics.values()]

	@property
	def display_names(self):
		"""Returns the display name identifiers of all registered metrics."""
		return [v['display_name'] for v in self.metrics.values()]


	@property
	def formatters(self):
		"""Returns the formatters for all metrics that have associated formatters."""
		return dict([(v['display_name'], v['formatter']) for k, v in self.metrics.items() if not v['formatter'] is None])

	#@property
	def val_dict(self, display_name = False, object = "metrics"):
		"""Returns dictionary of all registered values of object name or display_name as key.
		Params
        ------

       display_name: boolean, default = False
            If True, display_name of keys in dict. (default names)
        object: "cache" or "metrics", default = "metrics"
		"""
		if display_name: key_string = "display_name"
		else: key_string = "name"
		print("object dict: ", object)
		val_dict = dict([(self.__getattribute__(object)[key][key_string], self.__getattribute__(key)) for key in self.__getattribute__(object).keys() ])
		return val_dict

	def val_db(self, display_name = True):
		"""Returns dictionary of all registered values metrics to write in db."""
		if display_name: key_string = "display_name"
		else: key_string = "name"
		val_dict = dict([(self.metrics[key][key_string], self.__getattribute__(key)) for key in self.metrics.keys() if self.metrics[key]["write_db"] ])
		return val_dict


	def val_mail(self, display_name = True):
		"""Returns dictionary of all registered values metrics to write in mail."""
		if display_name: key_string = "display_name"
		else: key_string = "name"
		val_dict = dict([(self.metrics[key][key_string], self.__getattribute__(key)) for key in self.metrics.keys() if self.metrics[key]["write_mail"] ])
		return val_dict


	def to_dataframe(self, display_name = False, type = None):
		"""Returns pandas dataframe of all registered values metrics. """
		if type=="mail":
			self.df = pd.DataFrame(self.val_mail(display_name = display_name), index=[self.seqName])
		else:
			self.df = pd.DataFrame(self.val_dict(display_name = display_name), index=[self.seqName])
	def update_values(self, value_dict = None):
		"""Updates registered metrics with new values in value_dict. """
		if value_dict:
			for key, value in value_dict.items() :
				if hasattr(self, key):
					self.__setattr__(key, value)
				

	def print_type(self, object = "metrics"):
		"""Prints  variable type of registered metrics or caches. """
		print( "OBJECT " , object)
		val_dict = self.val_dict(object = object)
		for key, item in val_dict.items() :
			print("%s: %s; Shape: %s" %(key, type(item), np.shape(item)))

	def print_results(self):
		"""Prints metrics. """
		result_dict = self.val_dict()
		for key, item in result_dict.items():
			print(key)
			print("%s: %s" %(key, self.metrics[key]["formatter"](item)))


	def save_dict(self, path):
		"""Save value dict to path as pickle file."""
		with open(path, 'wb') as handle:
			pickle.dump(self.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)


	def compute_metrics_per_sequence(self):
		raise NotImplementedError


