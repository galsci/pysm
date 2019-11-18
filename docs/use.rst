Ho to use PySM
**************

Sky object
==========

The central object of PySM is the :class:`pysm.pysm.Sky` object. This is initialised using a dictionary in which we specify the required models::

  import pysm
  from pysm.nominal import models
  
  sky_config = {
  'dust' : [dust_pop_1, dust_pop_2, ...],
  'synchrotron' : [synch_pop_1, synch_pop_2, ...],
  'ame' : [ame_pop_1, ame_pop_2, ...],
  'freefree' : [ff_pop_1, ff_pop_2, ...],
  'cmb' : [cmb],
  }
  
The keys specify which components are present, and the items are lists of dictionaries which will be used to instantiate the relevant component class (:class:`pysm.components.Dust`, :class:`pysm.components.Synchrotron` etc). The number of dictionaries supplied for each component corresponds to the number of populations desired.
An individual component dictionary contains all the information specifying the emission model for that population of that component, e.g.::

  dust_pop_1 = {
  'model' : 'modified_black_body',
  'nu_0_I' : 545.,
  'nu_0_P' : 353.,
  'A_I' : read_map(template('dust_t_new.fits'), nside, field = 0),
  'A_Q' : read_map(template('dust_q_new.fits'), nside, field = 0),
  'A_U' : read_map(template('dust_u_new.fits'), nside, field = 0),
  'spectral_index' : read_map(template('dust_beta.fits'), nside = nside, field = 0),
  'temp' : read_map(template('dust_temp.fits'), nside, field = 0),
  'add_decorrelation' : False,
  }

PySM comes with many models pre-specified. The may be accessed by importing the relevant module::

  d5_config = models("d5", nside)
  s3_config = models("s3", nside)
  sky_config = {'dust' : d5_config, 'synchrotron' : s3_config}
  sky = pysm.Sky(sky_config)
  
One can then calculate the total emission, and individual component emission, at a single frequency or vector of frequencies::

  nu = np.array([10., 100., 500.])
  total_signal = sky.signal()(nu)
  dust_signal = sky.dust(nu)
  synchrotron_signal = sky.synchrotron(nu)

  import healpy
  import matplotlib.pyplot as plt
  
  hp.mollview(dust_signal[1, 0, :], title = "Dust T @ 100 GHz")
  hp.mollview(total_signal[0, 1, :], title = "Total Q @ 10 GHz")
  plt.show()


Instrument object
=================

Once a :class:`pysm.pysm.Sky` object has been instantiated we may then want to add instrumental effects. Currently PySM allows the integration of the signal over an arbitrary bandpass, smoothing with a Gaussian beam, and the addition of Gaussian white noise. These are all done using the :class:`pysm.pysm.Instrument` object::

  instrument = pysm.Instrument(instrument_config)

``instrument_config`` is a configuration dictionary specifying the instrument characteristics, for example::

  N_freqs = 20
  instrument_config = {
      'nside' : nside, 
      'frequencies' : np.logspace(1., 3., N_freqs), #Expected in GHz
      'use_smoothing' : True,
      'beams' : np.ones(N_freqs) * 70., #Expected in arcmin
      'add_noise' : True,
      'sens_I' : np.ones(N_freqs), #Expected in units uK_RJ
      'sens_P' : np.ones(N_freqs),
      'noise_seed' : 1234,
      'use_bandpass' : False,
      'output_units' : 'uK_RJ',
      'output_directory' : './',
      'output_prefix' : 'test',
  }
  
  instrument = pysm.Instrument(instrument_config)

We then use the :meth:`pysm.pysm.Instrument.observe()` method to observe the Sky we have already defined::
  
  instrument.observe(Sky)

This will write maps of (T, Q, U) as observed at the given frequencies with the given instrumental effects. 
  

Adding a new model
==================

PySM has been designed to make adding models as easy as possible. For example, say we have a new model that takes into account flattening of the synchrotron spectrum. We would need to edit only one part of the code, the :class:`pysm.components.Synchrotron` class. First we would write a function to represent our model::

  def model(nu):
      """Function to calculate synchrotron (T, Q, U)
      in flattening model.

      """
      # Do model calculations
      return np.array([T, Q, U])

Where ``nu`` is assumed to be a float, and ``np.array([T, Q, U])`` will have shape (3, Npix). We then insert this model into the Synchrotron class in components.py::

  class Synchrotron(object):
    ...
    ...
    ...
    """Note the name of the function returning our new model
    will be the name specified in the configuration
    dictionary.
    """
    def flattening(self):
        """Do any set up required for the model."""
	...
	...

	@Add_Decorrelation(self)
	@FloatOrArray
	def model(nu):
            # Do model calculations
	    return np.array([T, Q, U])
	    
	return model
	    
Where we have added the :func:`pysm.common.FloatOrArray` decorator to allow ``model`` input to be either a float or array, and we have added the option of frequency decorrelation through the :func:`pysm.components.Add_Decorrelation` decorator. If this model also requires some new parameter to be specified, ``flattening_parameter``, we must also add this as a property to the Synchrotron class::

  class Synchrotron(object):

  ...
  ...
  
  @property
  def Flattening_Parameter(self):
      try:
          return self.__flattening_parameter
      except AttributeError:
          print("Synchrotron attribute 'Flattening_Parameter' not set.")
	  sys.exit(1)

The final thing to do is to write a configuration dictionary for the new model::

  synch_flattening_conf = {
      'model' : 'flattening',
      'nu_0_I' : 0.408,
      'nu_0_P' : 23.,
      'A_I' : read_map(template('synch_t_new.fits'), nside, field = 0),
      'A_Q' : read_map(template('synch_q_new.fits'), nside, field = 0),
      'A_U' : read_map(template('synch_u_new.fits'), nside, field = 0),
      'flattening_parameter' : 0.4,
      'add_decorrelation' : True,    
  }

And then we can start using the new model in PySM::

  from pysm.nominal import models
  from new.models import synch_flattening_conf
  import pysm
  sky_config = {'dust' : models("d1", nside), 'synchrotron' : [synch_flattening_conf]}
  sky = pysm.Sky(sky_config)
  signal = sky.signal()
