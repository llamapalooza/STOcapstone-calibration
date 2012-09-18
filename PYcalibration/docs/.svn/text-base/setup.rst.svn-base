.. highlight:: console

=====================
Setup and Development
=====================

Requirements
------------

The code depends on the following packages:

* Python_ 2.5+ (< 3)
* NumPy_ 1.6+
* SciPy_ 0.10+
* OpenCV_ 2.3+ **with Python bindings**
* Cython_ 0.14+

To view and edit the interactive notebook files (``*.ipynb``), which contain some scratch
work and demos, you will also need:

* IPython_ 0.12+
* matplotlib_ 1.1.0+
* pyzmq_
* Tornado_

To compile this documentation, you will also need:

* Sphinx_

To easily run the tests (there aren't a lot of them), you might want:

* nose_

The best way to install Python packages is using the **easy_install** program included
with setuptools_. If you have a full-featured Python distribution, it might already be
installed on your system. Once you have it, all you need to do is run::

   $ sudo easy_install [package]

for each of the Python packages listed above (changing their names to lowercase).
Unfortunately, this doesn't work for OpenCV. You will also need the `ZeroMQ C library`_
before setting up pyzmq.

OpenCV
~~~~~~

This is the fun part. If you have OpenCV installed already, chances are, it is too old or
wasn't built with the Python bindings. Just to double-check, open the **python**
interpreter and enter:

.. code-block:: pycon

   >>> import cv2
   >>> quit()

If that works without producing an error, congratulations! Otherwise, you're SOL.

Once you've accepted your fate, try finding a more recent package for your system's
package manager. The Python bindings might also be included in a separate package.

After concluding your futile, hour-long search, download the latest source distribution
of OpenCV. While it's downloading, take a few shots and schedule an appointment with your
therapist. Next, unpack the distribution, but don't follow the `installation
instructions`_ quite yet. First make sure you have the Python header files. Many operating
systems distribute this as a separate package---for example, **python-dev**. Make sure the
version matches that of your default Python interpreter::

   $ python --version

Now follow the OpenCV installation instructions. After running **cmake**, you should see a
ton of information about what will be included---make sure the Python bindings are part of
it.

Since you now have spent several hours reading the documentation for CMake, sending
requests on three separate mailing lists, and manually editing the generated Makefiles,
give up and rewrite the three lines of code that depend on OpenCV. Everything you need
is probably already included in SciPy.

.. _Python: http://python.org/
.. _NumPy: http://numpy.scipy.org/
.. _SciPy: http://www.scipy.org/
.. _OpenCV: http://opencv.willowgarage.com/wiki/
.. _Cython: http://cython.org/
.. _IPython: http://ipython.org/
.. _matplotlib: http://matplotlib.sourceforge.net/
.. _pyzmq: http://zeromq.github.com/pyzmq/
.. _Tornado: http://www.tornadoweb.org/
.. _Sphinx: http://sphinx.pocoo.org/
.. _setuptools: http://pypi.python.org/pypi/setuptools
.. _ZeroMQ C library: http://www.zeromq.org/
.. _installation instructions: http://opencv.willowgarage.com/wiki/InstallGuide
.. _nose: http://readthedocs.org/docs/nose/en/latest/

Building
--------

Most of the code is pure Python and does not require any compilation. There is, however,
one module written in Cython which needs to be compiled before use. The easiest way to
accomplish this is to run::

    $ python setup.py build_ext --inplace

To build the documentation, run::

    $ make -C docs html

Running
-------

To use the interactive notebook, run::

   $ ipython notebook --pylab

To find point correspondences and an initial calibration estimate for a set of images,
run::

   $ bin/calibrate.py FOLDER > OUTPUT.corresp

The images folder should have the structure:

   FOLDER/
     ppm/
       XXXXR.ppm
       XXXXL.ppm
       ...
     corners/
       XXXXR.B.npy
       XXXXR.R.npy
       XXXXL.B.npy
       XXXXL.B.npy
       XXXXR.Bhom.txt
       XXXXR.Rhom.txt
       XXXXL.Bhom.txt
       XXXXL.Bhom.txt
       ...

where the :file:`*.npy` files are corner locations and the :file:`*.txt` files are
estimated homographies---both output from the corner detector.

Packaging
---------

This project includes a distutils_ setup file (``setup.py``) which makes it easy to
generate a source package. Simply run::

    $ python setup.py sdist

To create a tarball of the documentation, run::

    $ tar -czf calibration-docs.tar.gz -C docs/_build html

.. _distutils: http://docs.python.org/library/distutils.html

Testing
-------

Unit tests are few and far between, but if you feel a compulsion to do so, you can let
**nose** find and run them::

   $ nosetests

Troubleshooting
---------------

You're on your own. May the force be with you.