"""The IPython kernel implementation"""
import argparse
import ast
import asyncio
import binascii
import builtins
import copy
import getpass
import logging
import os
import re
import shlex
import signal
import sys
import threading
import time
import traceback
import typing as t
import uuid
from contextlib import contextmanager
from functools import partial

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from tornado import websocket

import comm
from IPython.core import release
from IPython.utils.tokenutil import line_at_cursor, token_at_cursor
from traitlets import Any, Bool, HasTraits, Instance, List, Type, observe, \
    observe_compat
from zmq.eventloop.zmqstream import ZMQStream

from .alpaca import deviceconnector
from .alpaca.utils import _to_png, unpack_Thonny_string
from .comm.comm import BaseComm
from .comm.manager import CommManager
from .compiler import XCachingCompiler
from .debugger import Debugger, _is_debugpy_available
from .eventloops import _use_appnope
from .kernelbase import Kernel as KernelBase
from .kernelbase import _accepts_cell_id
from .zmqshell import ZMQInteractiveShell

try:
    from IPython.core.interactiveshell import \
        _asyncio_runner  # type:ignore[attr-defined]
except ImportError:
    _asyncio_runner = None  # type:ignore

try:
    from IPython.core.completer import \
        provisionalcompleter as _provisionalcompleter
    from IPython.core.completer import \
        rectify_completions as _rectify_completions

    _use_experimental_60_completion = True
except ImportError:
    _use_experimental_60_completion = False

_EXPERIMENTAL_KEY_NAME = "_jupyter_types_experimental"


def _create_comm(*args, **kwargs):
    """Create a new Comm."""
    return BaseComm(*args, **kwargs)


# there can only be one comm manager in a alpaca_kernel process
_comm_lock = threading.Lock()
_comm_manager: t.Optional[CommManager] = None


def _get_comm_manager(*args, **kwargs):
    """Create a new CommManager."""
    global _comm_manager  # noqa
    if _comm_manager is None:
        with _comm_lock:
            if _comm_manager is None:
                _comm_manager = CommManager(*args, **kwargs)
    return _comm_manager


comm.create_comm = _create_comm
comm.get_comm_manager = _get_comm_manager

mpl.use('inline') # Force the matplotlib backend to be 'inline' by default

ap_plot = argparse.ArgumentParser(prog="%plot", add_help=False)
ap_plot.add_argument('--mode', type=str, default='matplotlib')
ap_plot.add_argument('--trigger_lvl', type=float, default=1.0)
ap_plot.add_argument('--type', type=str, default='RISE')
ap_plot.add_argument('--chan', type=int, default=1)

ap_bypass = argparse.ArgumentParser(prog="%python", add_help=False)

ap_serialconnect = argparse.ArgumentParser(prog="%serialconnect",
                                           add_help=False)
ap_serialconnect.add_argument('--raw', help='Just open connection',
                              action='store_true')
ap_serialconnect.add_argument('--port', type=str, default=0)
ap_serialconnect.add_argument('--baud', type=int, default=115200)
ap_serialconnect.add_argument('--verbose', action='store_true')

ap_socketconnect = argparse.ArgumentParser(prog="%socketconnect",
                                           add_help=False)
ap_socketconnect.add_argument('--raw', help='Just open connection',
                              action='store_true')
ap_socketconnect.add_argument('ipnumber', type=str)
ap_socketconnect.add_argument('portnumber', type=int)

ap_disconnect = argparse.ArgumentParser(prog="%disconnect", add_help=False)
ap_disconnect.add_argument('--raw',
                           help='Close connection without exiting paste mode',
                           action='store_true')

ap_websocketconnect = argparse.ArgumentParser(prog="%websocketconnect",
                                              add_help=False)
ap_websocketconnect.add_argument('--raw', help='Just open connection',
                                 action='store_true')
ap_websocketconnect.add_argument('websocketurl', type=str,
                                 default="ws://192.168.4.1:8266", nargs="?")
ap_websocketconnect.add_argument("--password", type=str)
ap_websocketconnect.add_argument('--verbose', action='store_true')

ap_writebytes = argparse.ArgumentParser(prog="%writebytes", add_help=False)
ap_writebytes.add_argument('--binary', '-b', action='store_true')
ap_writebytes.add_argument('--verbose', '-v', action='store_true')
ap_writebytes.add_argument('stringtosend', type=str)

ap_readbytes = argparse.ArgumentParser(prog="%readbytes", add_help=False)
ap_readbytes.add_argument('--binary', '-b', action='store_true')

ap_sendtofile = argparse.ArgumentParser(prog="%sendtofile",
                                        description="send a file to the microcontroller's file system",
                                        add_help=False)
ap_sendtofile.add_argument('--append', '-a', action='store_true')
ap_sendtofile.add_argument('--mkdir', '-d', action='store_true')
ap_sendtofile.add_argument('--binary', '-b', action='store_true')
ap_sendtofile.add_argument('--execute', '-x', action='store_true')
ap_sendtofile.add_argument('--source', help="source file", type=str,
                           default="<<cellcontents>>", nargs="?")
ap_sendtofile.add_argument('--quiet', '-q', action='store_true')
ap_sendtofile.add_argument('--QUIET', '-Q', action='store_true')
ap_sendtofile.add_argument('destinationfilename', type=str, nargs="?")

ap_ls = argparse.ArgumentParser(prog="%ls",
                                description="list directory of the microcontroller's file system",
                                add_help=False)
ap_ls.add_argument('--recurse', '-r', action='store_true')
ap_ls.add_argument('dirname', type=str, nargs="?")

ap_fetchfile = argparse.ArgumentParser(prog="%fetchfile",
                                       description="fetch a file from the microcontroller's file system",
                                       add_help=False)
ap_fetchfile.add_argument('--binary', '-b', action='store_true')
ap_fetchfile.add_argument('--print', '-p', action="store_true")
ap_fetchfile.add_argument('--load', '-l', action="store_true")
ap_fetchfile.add_argument('--quiet', '-q', action='store_true')
ap_fetchfile.add_argument('--QUIET', '-Q', action='store_true')
ap_fetchfile.add_argument('sourcefilename', type=str)
ap_fetchfile.add_argument('destinationfilename', type=str, nargs="?")

ap_mpycross = argparse.ArgumentParser(prog="%mpy-cross", add_help=False)
ap_mpycross.add_argument('--set-exe', type=str)
ap_mpycross.add_argument('pyfile', type=str, nargs="?")

ap_esptool = argparse.ArgumentParser(prog="%esptool", add_help=False)
ap_esptool.add_argument('--port', type=str, default=0)
ap_esptool.add_argument('espcommand', choices=['erase', 'esp32', 'esp8266'])
ap_esptool.add_argument('binfile', type=str, nargs="?")

ap_capture = argparse.ArgumentParser(prog="%capture",
                                     description="capture output printed by device and save to a file",
                                     add_help=False)
ap_capture.add_argument('--quiet', '-q', action='store_true')
ap_capture.add_argument('--QUIET', '-Q', action='store_true')
ap_capture.add_argument('outputfilename', type=str)

ap_writefilepc = argparse.ArgumentParser(prog="%%writefile",
                                         description="write contents of cell to file on PC",
                                         add_help=False)
ap_writefilepc.add_argument('--append', '-a', action='store_true')
ap_writefilepc.add_argument('--execute', '-x', action='store_true')
ap_writefilepc.add_argument('destinationfilename', type=str)


def parseap(ap, percentstringargs1):
    try:
        return ap.parse_known_args(percentstringargs1)[0]
    except SystemExit:  # argparse throws these because it assumes you only want to do the command line
        return None  # should be a default one


# --------------------- Plotting settings ----------------------------
DEFAULT_PLOT_MODE = 1  # 0 = no plot, 1 = matplotlib plot, 2 = live plot, 3 scope

# --------- Constants for plotting in matplotlib style ---------------
# Format for string is {dictionary of settings}[[x axis], [y axis]]
VALID_KEYS = ['color', 'linestyle', 'linewidth', 'marker', 'label']
# Format for attribute is ATTRIBUTE_PREFIXattribute(parameters)

VALID_ATTRIBUTES = {'legend': 'legend',
                    'axhline': 'axhline',
                    'axvline': 'axvline',
                    'hlines': 'hlines',
                    'vlines': 'vlines',
                    'grid': 'grid',
                    'xlim': 'set_xlim',
                    'ylim': 'set_ylim',
                    'xlabel': 'set_xlabel',
                    'ylabel': 'set_ylabel',
                    'title': 'set_title'}
# Key: accepted input, Value: function to run as ouput
ATTRIBUTE_PREFIX = '%matplotlib --'  # Prefix to recognize attribute
PLOT_PREFIX = '%matplotlibdata --'

# --------------------------------------------------------------------

serialtimeout = 0.5
serialtimeoutcount = 10

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
log = logging.getLogger('notebook')


def _get_plot_arguments(i, j, output):
    attribute = output[:i]
    # TODO: WHy are there two attributes here?
    filtered_attribute = VALID_ATTRIBUTES.get(attribute,
                                              attribute)
    args = output[i + 1:j]
    if args != '':
        args = args.split(', ')

        for i, arg in enumerate(args):
            if sum(cc.isalpha() for cc in arg) == 0:
                # Numbers
                args[i] = float(arg)
            else:
                args[i] = arg.replace('"', '').replace('\'', '')
    else:
        args = ()
    return args, filtered_attribute


class IPythonKernel(KernelBase):
    """The IPython Kernel class."""

    banner = "MicroPython Serializer for ALPACA"

    shell = Instance("IPython.core.interactiveshell.InteractiveShellABC",
                     allow_none=True)
    shell_class = Type(ZMQInteractiveShell)

    use_experimental_completions = Bool(
        True,
        help="Set this flag to False to deactivate the use of experimental IPython completion APIs.",
    ).tag(config=True)

    debugpy_stream = Instance(ZMQStream,
                              allow_none=True) if _is_debugpy_available else None

    user_module = Any()

    @observe("user_module")
    @observe_compat
    def _user_module_changed(self, change):
        if self.shell is not None:
            self.shell.user_module = change["new"]

    user_ns = Instance(dict, args=None, allow_none=True)

    @observe("user_ns")
    @observe_compat
    def _user_ns_changed(self, change):
        if self.shell is not None:
            self.shell.user_ns = change["new"]
            self.shell.init_user_ns()

    # A reference to the Python builtin 'raw_input' function.
    # (i.e., __builtin__.raw_input for Python 2.7, builtins.input for Python 3)
    _sys_raw_input = Any()
    _sys_eval_input = Any()

    def __init__(self, **kwargs):
        """Initialize the kernel."""
        super().__init__(**kwargs)

        # Initialize the Debugger
        self.ax = None
        self.fig = None
        self.use_micropython = True

        logging.info("Started Kernel")

        if _is_debugpy_available:
            self.debugger = Debugger(
                self.log,
                self.debugpy_stream,
                self._publish_debug_event,
                self.debug_shell_socket,
                self.session,
                self.debug_just_my_code,
            )

        # Initialize the InteractiveShell subclass
        self.shell = self.shell_class.instance(
            parent=self,
            profile_dir=self.profile_dir,
            user_module=self.user_module,
            user_ns=self.user_ns,
            kernel=self,
            compiler_class=XCachingCompiler,
        )
        self.shell.displayhook.session = self.session

        jupyter_session_name = os.environ.get('JPY_SESSION_NAME')
        if jupyter_session_name:
            self.shell.user_ns['__session__'] = jupyter_session_name

        self.shell.displayhook.pub_socket = self.iopub_socket
        self.shell.displayhook.topic = self._topic("execute_result")
        self.shell.display_pub.session = self.session
        self.shell.display_pub.pub_socket = self.iopub_socket

        self.comm_manager = comm.get_comm_manager()

        assert isinstance(self.comm_manager, HasTraits)
        self.shell.configurables.append(self.comm_manager)
        comm_msg_types = ["comm_open", "comm_msg", "comm_close"]
        for msg_type in comm_msg_types:
            self.shell_handlers[msg_type] = getattr(self.comm_manager, msg_type)

        if _use_appnope() and self._darwin_app_nap:
            # Disable app-nap as the kernel is not a gui but can have guis
            import appnope

            appnope.nope()

        self.silent = False
        self.dc = deviceconnector.DeviceConnector(self.sres, self.sresSYS,
                                                  self.sresPLOT)
        self.mpycrossexe = None

        self.srescapturemode = 0  # 0 none, 1 print lines, 2 print on-going line count (--quiet), 3 print only final line count (--QUIET)
        self.srescapturedoutputfile = None  # used by %capture command
        self.srescapturedlinecount = 0
        self.srescapturedlasttime = 0  # to control the frequency of capturing reported

        self.sresplotmode = DEFAULT_PLOT_MODE
        self.sres_trigger_lvl = 1.0
        self.sres_trig_RISE = True
        self.sres_trig_cldwn = True
        self.sres_has_trig = False
        self.sres_trig_chan = 1
        self.sresstartedplot = 0  #
        self.sresliveiteration = 0

    help_links = List(
        [
            {
                "text": "Python Reference",
                "url": "https://docs.python.org/%i.%i" % sys.version_info[:2],
            },
            {
                "text": "IPython Reference",
                "url": "https://ipython.org/documentation.html",
            },
            {
                "text": "NumPy Reference",
                "url": "https://docs.scipy.org/doc/numpy/reference/",
            },
            {
                "text": "SciPy Reference",
                "url": "https://docs.scipy.org/doc/scipy/reference/",
            },
            {
                "text": "Matplotlib Reference",
                "url": "https://matplotlib.org/contents.html",
            },
            {
                "text": "SymPy Reference",
                "url": "http://docs.sympy.org/latest/index.html",
            },
            {
                "text": "pandas Reference",
                "url": "https://pandas.pydata.org/pandas-docs/stable/",
            },
        ]
    ).tag(config=True)

    # Kernel info fields
    implementation = 'alpaca_kernel_v2'
    implementation_version = release.version
    language_info = {
        "name": "python",
        "version": sys.version.split()[0],
        "mimetype": "text/x-python",
        "codemirror_mode": {"name": "ipython", "version": sys.version_info[0]},
        "pygments_lexer": "ipython%d" % 3,
        "nbconvert_exporter": "python",
        "file_extension": ".py",
    }

    def dispatch_debugpy(self, msg):
        if _is_debugpy_available:
            # The first frame is the socket id, we can drop it
            frame = msg[1].bytes.decode("utf-8")
            self.log.debug("Debugpy received: %s", frame)
            self.debugger.tcp_client.receive_dap_frame(frame)

    # @property
    # def banner(self):
    #     return self.shell.banner

    async def poll_stopped_queue(self):
        """Poll the stopped queue."""
        while True:
            await self.debugger.handle_stopped_event()

    def start(self):
        """Start the kernel."""
        self.shell.exit_now = False
        if self.debugpy_stream is None:
            self.log.warning(
                "debugpy_stream undefined, debugging will not be enabled")
        else:
            self.debugpy_stream.on_recv(self.dispatch_debugpy, copy=False)
        super().start()
        if self.debugpy_stream:
            asyncio.run_coroutine_threadsafe(
                self.poll_stopped_queue(),
                self.control_thread.io_loop.asyncio_loop
            )

    def set_parent(self, ident, parent, channel="shell"):
        """Overridden from parent to tell the display hook and output streams
        about the parent message.
        """
        super().set_parent(ident, parent, channel)
        if channel == "shell":
            self.shell.set_parent(parent)

    def init_metadata(self, parent):
        """Initialize metadata.

        Run at the beginning of each execution request.
        """
        md = super().init_metadata(parent)
        # FIXME: remove deprecated ipyparallel-specific code
        # This is required for ipyparallel < 5.0
        md.update(
            {
                "dependencies_met": True,
                "engine": self.ident,
            }
        )
        return md

    def finish_metadata(self, parent, metadata, reply_content):
        """Finish populating metadata.

        Run after completing an execution request.
        """
        # FIXME: remove deprecated ipyparallel-specific code
        # This is required by ipyparallel < 5.0
        metadata["status"] = reply_content["status"]
        if reply_content["status"] == "error" and reply_content[
            "ename"] == "UnmetDependency":
            metadata["dependencies_met"] = False

        return metadata

    def _forward_input(self, allow_stdin=False):
        """Forward raw_input and getpass to the current frontend.

        via input_request
        """
        self._allow_stdin = allow_stdin

        self._sys_raw_input = builtins.input
        builtins.input = self.raw_input

        self._save_getpass = getpass.getpass
        getpass.getpass = self.getpass

    def _restore_input(self):
        """Restore raw_input, getpass"""
        builtins.input = self._sys_raw_input

        getpass.getpass = self._save_getpass

    @property
    def execution_count(self):
        return self.shell.execution_count

    @execution_count.setter
    def execution_count(self, value):
        # Ignore the incrementing done by KernelBase, in favour of our shell's
        # execution counter.
        pass

    @contextmanager
    def _cancel_on_sigint(self, future):
        """ContextManager for capturing SIGINT and cancelling a future

        SIGINT raises in the event loop when running async code,
        but we want it to halt a coroutine.

        Ideally, it would raise KeyboardInterrupt,
        but this turns it into a CancelledError.
        At least it gets a decent traceback to the user.
        """
        sigint_future: asyncio.Future[int] = asyncio.Future()

        # whichever future finishes first,
        # cancel the other one
        def cancel_unless_done(f, _ignored):
            if f.cancelled() or f.done():
                return
            f.cancel()

        # when sigint finishes,
        # abort the coroutine with CancelledError
        sigint_future.add_done_callback(partial(cancel_unless_done, future))
        # when the main future finishes,
        # stop watching for SIGINT events
        future.add_done_callback(partial(cancel_unless_done, sigint_future))

        def handle_sigint(*args):
            def set_sigint_result():
                if sigint_future.cancelled() or sigint_future.done():
                    return
                sigint_future.set_result(1)

            # use add_callback for thread safety
            self.io_loop.add_callback(set_sigint_result)

        # set the custom sigint hander during this context
        save_sigint = signal.signal(signal.SIGINT, handle_sigint)
        try:
            yield
        finally:
            # restore the previous sigint handler
            signal.signal(signal.SIGINT, save_sigint)

    def sresSYS(self, output, clear_output=False):  # system call
        self.sres(output, asciigraphicscode=34, clear_output=clear_output)

    def sres(self, output, asciigraphicscode=None, n04count=0,
             clear_output=False):
        if self.silent:
            return

        # logging.info(output)

        if (self.srescapturedoutputfile
                and (n04count == 0)
                and not asciigraphicscode):
            self.srescapturedoutputfile.write(output)
            self.srescapturedlinecount += len(output.split("\n")) - 1
            # 0 none, 1 print lines, 2 print on-going line count (--quiet),
            # 3 print only final line count (--QUIET)
            if self.srescapturemode == 3:
                return

            # changes the printing out to a lines captured
            # statement every 1 second.
            if self.srescapturemode == 2:
                # (allow stderrors to drop through to normal printing
                srescapturedtime = time.time()
                # update no more frequently than once a second
                if srescapturedtime < self.srescapturedlasttime + 1:
                    return
                self.srescapturedlasttime = srescapturedtime
                clear_output = True
                output = "{} lines captured".format(self.srescapturedlinecount)

        if clear_output:  # used when updating lines printed
            self.send_response(self.iopub_socket, 'clear_output',
                               {"wait": True})
        if asciigraphicscode:
            output = "\x1b[{}m{}\x1b[0m".format(asciigraphicscode, output)

        stream_content = {'name': ("stdout" if n04count == 0 else "stderr"),
                          'text': output}

        self.send_response(self.iopub_socket, 'stream', stream_content)

    def sresPLOT(self, output: str, asciigraphicscode=None, n04count=0,
                 clear_output=False):

        if self.silent:
            return

        # if output is None or output == '':
        #    return

        if self.sresplotmode == 0:
            # Plotting on but no plot commands used in code
            self.sres(output)
            return

        if self.sresplotmode == 1:  # matplotlib-esque (array) plotting

            if output == ATTRIBUTE_PREFIX + "--update" and self.sresstartedplot:
                self.sendPLOT()
                return None

            # ------------------- PLOT SETTINGS ---------------------
            # TODO: Make this more robust against LaTeX syntax
            #  containing {} in labels.
            if output is not None and ATTRIBUTE_PREFIX in output:
                # User is trying change the plot settings
                try:
                    if self.ax is not None:  # If plot was made
                        output = output.replace(ATTRIBUTE_PREFIX, '')
                        args_start = output.find('(')
                        args_end = output.find(
                            '{') - 2 if '{' in output else output.find(')')

                        args, filtered_attribute = _get_plot_arguments(
                            args_start, args_end, output)

                        kwargs_start = args_end + 2

                        kwargs = self._get_plot_kwargs(kwargs_start, output)
                    else:
                        return None
                except (AttributeError, SyntaxError, ValueError) as e:
                    # Catch formatting errors
                    self.sres(output)
                    return None

                try:
                    getattr(self.ax, filtered_attribute)(*args,
                                                         **kwargs)  # Run command
                    logging.debug(f'Plot setting {filtered_attribute} changed')
                    return None
                except Exception:
                    # Pass plotting errors to user
                    tb = traceback.format_exc()
                    self.sres(tb, n04count=1)
                    return None

            # ----------- PLOT DATA -------------------
            elif output is not None and PLOT_PREFIX in output:

                try:  # Normal plot, no settings
                    output = output.replace(PLOT_PREFIX, '')
                    kwargs_end = output.rfind('}')
                    settings = output[:kwargs_end + 1]
                    data = output[kwargs_end + 1:]

                    settings = ast.literal_eval(settings)

                    self._get_xy_data(data)

                except Exception as e:
                    # Incorrect formatting, this should not happen when using
                    # The plotting module for the ALPACA
                    self.sres(output)

                    return None

                # the data is good and plotting can commence
                if not self.sresstartedplot:
                    logging.debug('Created a new plot')
                    self.sresPLOTcreator()

                fmt, kwargs = self._get_format_string(settings)

                try:
                    self.ax.plot(self.xx, self.yy, fmt, **kwargs)
                except Exception:
                    # Pass plotting errors to user
                    tb = traceback.format_exc()
                    self.sres(tb, n04count=1)
                    return None
                return None

            else:  # Not something to plot, just print
                self.sres(output)

        elif self.sresplotmode in (2, 3, 4):  # Thonny-eqsue plotting or
            # scope-esque plotting
            # format print("Random walk:", p1, " just random:", p2)
            try:

                if sum(cc.isnumeric() for cc in output) == 0:
                    # Plain text print statement
                    self.sres(output)
                    return

                data = unpack_Thonny_string(output)

            except ValueError as e:
                self.sres(output)
                logging.exception(e)
                return

            # the data is good and plotting can commence

            if not self.sresstartedplot:
                # or len(data) != self.number_lines: # (re)Instantiation
                try:
                    # if not self.sresstartedplot:
                    self.sresPLOTcreator()
                    self.sresstartedplottime = time.time()

                    self.number_lines = len(data)

                    self.yy_minimum = 1_000
                    self.yy_maximum = -1_000

                    self.xx_minimum = 1_000
                    self.xx_maximum = -1_000

                    # Sanitize trigger channel input with knowledge of amount
                    # of data
                    if self.sres_trig_chan < 1:
                        self.sres_trig_chan = 1
                    elif self.sres_trig_chan > self.number_lines:
                        self.sres_trig_chan = self.number_lines

                    self.yy = np.zeros((0, self.number_lines))
                    self.xx = np.zeros(0)
                    self.lines = self.ax.plot(self.xx, self.yy)
                    for args_start, line in enumerate(self.lines):
                        line.set_label(list(data.keys())[args_start])

                    self.ax.legend(loc='upper center', ncol=3)
                    self.ax.grid()
                    self.ax.set_xlabel("Time [s]")
                except Exception as e:
                    self.sresstartedplot = 0
                    self.sres(output, n04count=n04count)
                    logging.exception(e)
                    return None

            try:
                if self.sresplotmode == 4:  # EEG style (auto-scroll)
                    if self.sresliveiteration < 100:
                        self.yy = np.append(
                            self.yy,
                            [list(data.values())], axis=0)
                        self.xx = np.append(
                            self.xx,
                            time.time() - self.sresstartedplottime)
                    else:
                        self.yy = np.append(
                            self.yy[1:, :],
                            [list(data.values())], axis=0)
                        self.xx = np.append(
                            self.xx[1:],
                            time.time() - self.sresstartedplottime)

                elif self.sresplotmode == 2:  # Live plot (no scroll)
                    self.yy = np.append(
                        self.yy,
                        [list(data.values())], axis=0)
                    self.xx = np.append(
                        self.xx,
                        time.time() - self.sresstartedplottime)

                else:
                    data_l = list(data.values())
                    value = data_l[self.sres_trig_chan - 1]
                    triggered_now = False
                    if self.sres_trig_RISE:
                        if (not self.sres_trig_cldwn
                                and value > self.sres_trigger_lvl):
                            triggered_now = True
                        if self.sres_trig_cldwn and value < self.sres_trigger_lvl:
                            self.sres_trig_cldwn = False
                    else:
                        if not self.sres_trig_cldwn and value < self.sres_trigger_lvl:
                            triggered_now = True
                        if self.sres_trig_cldwn and value > self.sres_trigger_lvl:
                            self.sres_trig_cldwn = False

                    if triggered_now:
                        self.sres_has_trig = True
                        self.sres_trig_cldwn = True
                        self.sresstartedplottime = time.time()
                        self.yy = np.zeros((0, self.number_lines))
                        self.xx = np.zeros(0)

                    self.yy = np.append(self.yy, [data_l], axis=0)
                    self.xx = np.append(self.xx,
                                        time.time() - self.sresstartedplottime)

                # self.ax.cla() # Clear
                for args_start, line in enumerate(self.lines):
                    line.set_xdata(self.xx)
                    line.set_ydata(self.yy[:, args_start])

                # self.ax.autoscale()
                yy_minimum = np.amin(self.yy)
                yy_maximum = np.amax(self.yy)
                xx_minimum = np.amin(self.xx)
                xx_maximum = np.amax(self.xx)
                if self.sresplotmode == 2:
                    yy_edge_size = (yy_maximum - yy_minimum) / 10 + 0.001
                    xx_edge_size = (xx_maximum - xx_minimum) / 10 + 0.001

                    self.ax.set_ylim(yy_minimum - yy_edge_size,
                                     yy_maximum + yy_edge_size * 2)  # Extra space for legend
                    self.ax.set_xlim(xx_minimum - xx_edge_size,
                                     xx_maximum + xx_edge_size)
                else:
                    if ((yy_minimum < self.yy_minimum)
                            or (yy_maximum > self.yy_maximum)
                            or (xx_minimum < self.xx_minimum)
                            or (xx_maximum > self.xx_maximum)):
                        self.yy_minimum = min(yy_minimum, self.yy_minimum)
                        self.yy_maximum = max(yy_maximum, self.yy_maximum)
                        self.xx_minimum = min(xx_minimum, self.xx_minimum)
                        self.xx_maximum = max(xx_maximum, self.xx_maximum)

                        yy_edge_size = (self.yy_maximum - self.yy_minimum) / 10
                        xx_edge_size = (self.xx_maximum - self.xx_minimum) / 10

                        self.ax.set_ylim(self.yy_minimum - yy_edge_size,
                                         self.yy_maximum + yy_edge_size * 2)  # Extra space for legend
                        self.ax.set_xlim(self.xx_minimum - xx_edge_size,
                                         self.xx_maximum + xx_edge_size)
                # self.ax.plot(self.xx, self.yy, label =  # Plot

                if self.sresliveiteration:
                     self.sendPLOT(
                        update_id=self.plot_uuid)  # Use old plot and display
                else:
                    self.plot_uuid = self.sendPLOT()  # Create new plot and store UUID

                self.sresliveiteration += 1
            except Exception as e:
                self.sres(output, n04count=n04count)
                tb = traceback.format_exc()
                logging.exception(tb)
                return None
            return None

    def _get_format_string(self, settings):
        # default value for [fmt]
        fmt = settings.pop('fmt', '')
        kwargs = {}
        for key, value in settings.items():
            if key in VALID_KEYS:
                kwargs[key] = value
        return fmt, kwargs

    def _get_xy_data(self, data):
        logging.debug(f"{len(data)}DATASTART{data}DATAEND")
        args_start = data.find('], [')

        xx_hex_data, yy_hex_data = (
            data[2: args_start], data[args_start + 4:data.rfind(']') - 1])

        logging.debug(f"{len(xx_hex_data)}XSTART{xx_hex_data}XEND")
        logging.debug(f"{len(yy_hex_data)}YSTART{yy_hex_data}YEND")

        xx_returned_data = bytearray(binascii.unhexlify(xx_hex_data))
        yy_returned_data = bytearray(binascii.unhexlify(yy_hex_data))
        try:
            yy_shape_string = data[data.rfind('('):]
            yy_shape = ast.literal_eval(yy_shape_string)
        except Exception as e:
            raise RuntimeError(
                f'Couldn\'t read shape from string: {yy_shape_string}') from e
        self.xx = np.frombuffer(xx_returned_data, dtype=np.float32)
        self.yy = np.frombuffer(yy_returned_data, dtype='f').reshape(yy_shape)

    def _get_plot_kwargs(self, kwargs_start, output):
        if '{' in output and '{}' not in output:  # read kwargs
            kwargs = output[kwargs_start:output.find(')')]
            kwargs = ast.literal_eval(kwargs)
        else:
            kwargs = {}
        return kwargs

    def sresPLOTcreator(self):
        # We create the plot with matplotlib.
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
        self.sresstartedplot = True

    def sresPLOTkiller(self):
        self.sresplotmode = DEFAULT_PLOT_MODE  # Reset plot
        self.sresstartedplot = False
        self.fig, self.ax = (None, None)
        self.sresliveiteration = 0

    def sendPLOT(self, update_id=None) -> int:
        if self.sresstartedplot == 0:
            return 0

        # We create a PNG out of this plot.
        png = _to_png(self.fig)

        if update_id == None:
            plot_uuid = uuid.uuid1()  # When creating a new plot
            update = False
        else:
            plot_uuid = update_id
            update = True

        # We send the standard output to the
        # client.
        # self.send_response(
        #    self.iopub_socket,
        #    'stream', {
        #        'name': 'stdout',
        #        'data': ('Plotting {n} '
        #                'data points'). \
        #                format(n=self.xx.size)})

        # We prepare the response with our rich
        # data (the plot).
        content = {
            # 'source': 'kernel',

            # This dictionary may contain
            # different MIME representations of
            # the output.
            'data': {
                'image/png': png
            },

            # We can specify the image size
            # in the metadata field.
            'metadata': {
                'image/png': {
                    'width': 600,
                    'height': 400
                }
            },

            'transient': {
                'display_id': str(plot_uuid)
            }
        }

        # We send the display_data message with
        # the contents.
        if update:

            # self.send_response(self.iopub_socket,
            #    'clear_output', {'wait' : True})

            self.send_response(self.iopub_socket,
                               'update_display_data', content)

            # logging.debug(f'Updated display data')
        else:  # creating new plot
            self.send_response(self.iopub_socket,
                               'display_data', content)

            return plot_uuid  # Return new UUID for futre reference
            # logging.debug(f'Created new display data')

        return 1

    def interpretpercentline(self, percentline, cellcontents):
        try:
            percentstringargs = shlex.split(percentline)
        except ValueError as e:
            self.sres("\n\n***Bad percentcommand [%s]\n" % str(e), 31)
            self.sres(percentline)
            return None

        percentcommand = percentstringargs[0]

        if percentcommand == ap_bypass.prog:
            self.use_micropython = False

        if percentcommand == ap_serialconnect.prog:
            apargs = parseap(ap_serialconnect, percentstringargs[1:])

            self.dc.disconnect(apargs.verbose)
            self.dc.serialconnect(apargs.port, apargs.baud, apargs.verbose)
            if self.dc.workingserial:
                if not apargs.raw:
                    if self.dc.enterpastemode(verbose=apargs.verbose):
                        self.sresSYS("Ready.\n")
                    else:
                        self.sres("Disconnecting [paste mode not working]\n",
                                  31)
                        self.dc.disconnect(verbose=apargs.verbose)
                        self.sresSYS("  (You may need to reset the device)")
                        cellcontents = ""
            else:
                cellcontents = ""
            return cellcontents.strip() and cellcontents or None

        if percentcommand == ap_websocketconnect.prog:
            apargs = parseap(ap_websocketconnect, percentstringargs[1:])
            if apargs.password is None and not apargs.raw:
                self.sres(ap_websocketconnect.format_help())
                return None
            self.dc.websocketconnect(apargs.websocketurl)
            if self.dc.workingwebsocket:
                self.sresSYS("** WebSocket connected **\n", 32)
                if not apargs.raw:
                    pline = self.dc.workingwebsocket.recv()
                    self.sres(pline)
                    if pline == 'Password: ' and apargs.password is not None:
                        self.dc.workingwebsocket.send(apargs.password)
                        self.dc.workingwebsocket.send("\r\n")
                        res = self.dc.workingserialreadall()
                        self.sres(res)  # '\r\nWebREPL connected\r\n>>> '
                        if not apargs.raw:
                            if self.dc.enterpastemode(apargs.verbose):
                                self.sresSYS("Ready.\n")
                            else:
                                self.sres(
                                    "Disconnecting [paste mode not working]\n",
                                    31)
                                self.dc.disconnect(verbose=apargs.verbose)
                                self.sres(
                                    "  (You may need to reset the device)")
                                cellcontents = ""
            else:
                cellcontents = ""
            return cellcontents.strip() and cellcontents or None

        # this is the direct socket kind, not attached to a webrepl
        if percentcommand == ap_socketconnect.prog:
            apargs = parseap(ap_socketconnect, percentstringargs[1:])
            self.dc.socketconnect(apargs.ipnumber, apargs.portnumber)
            if self.dc.workingsocket:
                self.sres("\n ** Socket connected **\n\n", 32)
                if apargs.verbose:
                    self.sres(str(self.dc.workingsocket))
                self.sres("\n")
                # if not apargs.raw:
                #    self.dc.enterpastemode()
            return cellcontents.strip() and cellcontents or None

        if percentcommand == ap_esptool.prog:
            apargs = parseap(ap_esptool, percentstringargs[1:])
            if apargs and (apargs.espcommand == "erase" or apargs.binfile):
                self.dc.esptool(apargs.espcommand, apargs.port, apargs.binfile)
            else:
                self.sres(ap_esptool.format_help())
                self.sres(
                    "Please download the bin file from https://micropython.org/download/#{}".format(
                        apargs.espcommand if apargs else ""))
            return cellcontents.strip() and cellcontents or None

        if percentcommand == ap_writefilepc.prog:
            apargs = parseap(ap_writefilepc, percentstringargs[1:])
            if apargs:
                if apargs.append:
                    self.sres("Appending to {}\n\n".format(
                        apargs.destinationfilename), asciigraphicscode=32)
                    fout = open(apargs.destinationfilename, ("a"))
                    fout.write("\n")
                else:
                    self.sres(
                        "Writing {}\n\n".format(apargs.destinationfilename),
                        asciigraphicscode=32)
                    fout = open(apargs.destinationfilename, ("w"))

                fout.write(cellcontents)
                fout.close()
            else:
                self.sres(ap_writefilepc.format_help())
            if not apargs.execute:
                return None
            return cellcontents  # should add in some blank lines at top to get errors right

        if percentcommand == "%mpy-cross":
            apargs = parseap(ap_mpycross, percentstringargs[1:])
            if apargs and apargs.set_exe:
                self.mpycrossexe = apargs.set_exe
            elif apargs.pyfile:
                if self.mpycrossexe:
                    self.dc.mpycross(self.mpycrossexe, apargs.pyfile)
                else:
                    self.sres("Cross compiler executable not yet set\n", 31)
                    self.sres(
                        "try: %mpy-cross --set-exe /home/julian/extrepositories/micropython/mpy-cross/mpy-cross\n")
                if self.mpycrossexe:
                    self.mpycrossexe = "/home/julian/extrepositories/micropython/mpy-cross/mpy-cross"
            else:
                self.sres(ap_mpycross.format_help())
            return cellcontents.strip() and cellcontents or None

        if percentcommand == "%comment":
            self.sres(" ".join(percentstringargs[1:]), asciigraphicscode=32)
            return cellcontents.strip() and cellcontents or None

        if percentcommand == "%lsmagic":
            self.sres(re.sub("usage: ", "", ap_capture.format_usage()))
            self.sres("    records output to a file\n\n")
            self.sres("%comment\n    print this into output\n\n")
            self.sres(re.sub("usage: ", "", ap_disconnect.format_usage()))
            self.sres("    disconnects from web/serial connection\n\n")
            self.sres(re.sub("usage: ", "", ap_esptool.format_usage()))
            self.sres("    commands for flashing your esp-device\n\n")
            self.sres(re.sub("usage: ", "", ap_fetchfile.format_usage()))
            self.sres("    fetch and save a file from the device\n\n")
            self.sres(re.sub("usage: ", "", ap_ls.format_usage()))
            self.sres("    list files on the device\n\n")
            self.sres("%lsmagic\n    list magic commands\n\n")
            self.sres(re.sub("usage: ", "", ap_mpycross.format_usage()))
            self.sres("    cross-compile a .py file to a .mpy file\n\n")
            self.sres(re.sub("usage: ", "", ap_readbytes.format_usage()))
            self.sres("    does serial.read_all()\n\n")
            self.sres("%rebootdevice\n    reboots device\n\n")
            self.sres(re.sub("usage: ", "", ap_sendtofile.format_usage()))
            self.sres(
                "    send cell contents or file/direcectory to the device\n\n")
            self.sres(re.sub("usage: ", "", ap_serialconnect.format_usage()))
            self.sres("    connects to a device over USB wire\n\n")
            self.sres(re.sub("usage: ", "", ap_socketconnect.format_usage()))
            self.sres("    connects to a socket of a device over wifi\n\n")
            self.sres(
                "%suppressendcode\n    doesn't send x04 or wait to read after sending the contents of the cell\n")
            self.sres(
                "  (assists for debugging using %writebytes and %readbytes)\n\n")
            self.sres(re.sub("usage: ", "", ap_websocketconnect.format_usage()))
            self.sres(
                "    connects to the webREPL websocket of an ESP8266 over wifi\n")
            self.sres(
                "    websocketurl defaults to ws://192.168.4.1:8266 but be sure to be connected\n\n")
            self.sres(re.sub("usage: ", "", ap_writebytes.format_usage()))
            self.sres(
                "    does serial.write() of the python quoted string given\n\n")
            self.sres(re.sub("usage: ", "", ap_writefilepc.format_usage()))
            self.sres("    write contents of cell to a file\n\n")

            return None

        if percentcommand == ap_disconnect.prog:
            apargs = parseap(ap_disconnect, percentstringargs[1:])
            self.dc.disconnect(raw=apargs.raw, verbose=True)
            return None

        # remaining commands require a connection
        if not self.dc.serialexists():
            return cellcontents

        if percentcommand == ap_plot.prog:
            apargs = parseap(ap_plot, percentstringargs[1:])
            if apargs.mode == 'matplotlib':
                self.sresplotmode = 1  # matplotlib-esque (array) plotting
            elif apargs.mode == 'live':
                self.sresplotmode = 2  # live plotting

            elif apargs.mode == 'livescroll':
                self.sresplotmode = 4  # live plotting with auto-scroll

            elif apargs.mode == 'scope':
                self.sresplotmode = 3  # scope-style
                self.sres_trigger_lvl = apargs.trigger_lvl
                self.sres_trig_RISE = True if apargs.type == 'RISE' else False
                self.sres_trig_chan = apargs.chan

            elif apargs.mode == 'none':
                self.sresplotmode = 0
            else:
                self.sresplotmode = DEFAULT_PLOT_MODE
            return cellcontents

        if percentcommand == ap_capture.prog:
            apargs = parseap(ap_capture, percentstringargs[1:])
            if apargs:
                self.sres("Writing output to file {}\n\n".format(
                    apargs.outputfilename), asciigraphicscode=32)
                self.srescapturedoutputfile = open(apargs.outputfilename, "w")
                self.srescapturemode = (
                    3 if apargs.QUIET else (2 if apargs.quiet else 1))
                self.srescapturedlinecount = 0
            else:
                self.sres(ap_capture.format_help())
            return cellcontents

        if percentcommand == ap_writebytes.prog:
            # (not effectively using the --binary setting)
            apargs = parseap(ap_writebytes, percentstringargs[1:])
            if apargs:
                bytestosend = apargs.stringtosend.encode().decode(
                    "unicode_escape").encode()
                res = self.dc.writebytes(bytestosend)
                if apargs.verbose:
                    self.sres(res, asciigraphicscode=34)
            else:
                self.sres(ap_writebytes.format_help())
            return cellcontents.strip() and cellcontents or None

        if percentcommand == ap_readbytes.prog:
            # (not effectively using the --binary setting)
            apargs = parseap(ap_readbytes, percentstringargs[1:])
            time.sleep(
                0.1)  # just give it a moment if running on from a series of values (could use an --expect keyword)
            l = self.dc.workingserialreadall()
            if apargs.binary:
                self.sres(repr(l))
            elif type(l) == bytes:
                self.sres(l.decode(errors="ignore"))
            else:
                self.sres(l)  # strings come back from webrepl
            return cellcontents.strip() and cellcontents or None

        if percentcommand == "%rebootdevice":
            self.dc.sendrebootmessage()
            self.dc.enterpastemode()
            return cellcontents.strip() and cellcontents or None

        if percentcommand == "%reboot":
            self.sres("Did you mean %rebootdevice?\n", 31)
            return None

        if percentcommand == "%%writetofile" or percentcommand == "%writefile":
            self.sres("Did you mean %%writefile?\n", 31)
            return None

        if percentcommand == "%serialdisconnect":
            self.sres("Did you mean %disconnect?\n", 31)
            return None

        if percentcommand == "%sendbytes":
            self.sres("Did you mean %writebytes?\n", 31)
            return None

        if percentcommand == "%reboot":
            self.sres("Did you mean %rebootdevice?\n", 31)
            return None

        if percentcommand in ("%savetofile", "%savefile", "%sendfile"):
            self.sres("Did you mean to write %sendtofile?\n", 31)
            return None

        if percentcommand in ("%readfile", "%fetchfromfile"):
            self.sres("Did you mean to write %fetchfile?\n", 31)
            return None

        if percentcommand == ap_fetchfile.prog:
            apargs = parseap(ap_fetchfile, percentstringargs[1:])
            if apargs:
                fetchedcontents = self.dc.fetchfile(apargs.sourcefilename,
                                                    apargs.binary, apargs.quiet)
                if apargs.print:
                    self.sres(fetchedcontents.decode() if type(
                        fetchedcontents) == bytes else fetchedcontents,
                              clear_output=True)

                if (apargs.destinationfilename or (
                        not apargs.print and not apargs.load)) and fetchedcontents:
                    dstfile = apargs.destinationfilename or os.path.basename(
                        apargs.sourcefilename)
                    self.sres("Saving file to {}".format(repr(dstfile)))
                    fout = open(dstfile, "wb" if apargs.binary else "w")
                    fout.write(fetchedcontents)
                    fout.close()

                if apargs.load:
                    fcontents = fetchedcontents.decode() if type(
                        fetchedcontents) == bytes else fetchedcontents
                    if not apargs.quiet:
                        fcontents = "#%s\n\n%s" % (
                        " ".join(percentstringargs), fcontents)
                    set_next_input_payload = {"source": "set_next_input",
                                              "text": fcontents,
                                              "replace": True}
                    return set_next_input_payload

            else:
                self.sres(ap_fetchfile.format_help())
            return None

        if percentcommand == ap_ls.prog:
            apargs = parseap(ap_ls, percentstringargs[1:])
            if apargs:
                self.dc.listdir(apargs.dirname or "", apargs.recurse)
            else:
                self.sres(ap_ls.format_help())
            return None

        if percentcommand == ap_sendtofile.prog:
            apargs = parseap(ap_sendtofile, percentstringargs[1:])
            if apargs and not (
                    apargs.source == "<<cellcontents>>" and not apargs.destinationfilename) and (
                    apargs.source != None):

                destfn = apargs.destinationfilename

                def sendtofile(filename, contents):
                    self.dc.sendtofile(filename, apargs.mkdir, apargs.append,
                                       apargs.binary, apargs.quiet, contents)

                if apargs.source == "<<cellcontents>>":
                    filecontents = cellcontents
                    if not apargs.execute:
                        cellcontents = None
                    sendtofile(destfn, filecontents)

                else:
                    mode = "rb" if apargs.binary else "r"
                    if not destfn:
                        destfn = os.path.basename(apargs.source)
                    elif destfn[-1] == "/":
                        destfn += os.path.basename(apargs.source)

                    if os.path.isfile(apargs.source):
                        filecontents = open(apargs.source, mode).read()
                        if apargs.execute:
                            self.sres("Cannot excecute sourced file\n", 31)
                        sendtofile(destfn, filecontents)

                    elif os.path.isdir(apargs.source):
                        if apargs.execute:
                            self.sres("Cannot excecute folder\n", 31)
                        for root, dirs, files in os.walk(apargs.source):
                            for fn in files:
                                skip = False
                                fp = os.path.join(root, fn)
                                relpath = os.path.relpath(fp, apargs.source)
                                if relpath.endswith('.py'):
                                    # Check for compiled copy, skip py if exists
                                    if os.path.exists(fp[:-3] + '.mpy'):
                                        skip = True
                                if not skip:
                                    destpath = os.path.join(destfn,
                                                            relpath).replace(
                                        '\\', '/')
                                    filecontents = open(os.path.join(root, fn),
                                                        mode).read()
                                    sendtofile(destpath, filecontents)
            else:
                self.sres(ap_sendtofile.format_help())
            return cellcontents  # allows for repeat %sendtofile in same cell

        self.sres("Unrecognized percentline {}\n".format([percentline]), 31)
        return cellcontents

    def runnormalcell(self, cellcontents, bsuppressendcode):
        cmdlines = cellcontents.splitlines(True)
        r = self.dc.workingserialreadall()
        if r:
            self.sres('[priorstuff] ')
            self.sres(str(r))

        for line in cmdlines:
            if line:
                if line[-2:] == '\r\n':
                    line = line[:-2]
                elif line[-1] == '\n':
                    line = line[:-1]
                self.dc.writeline(line)
                r = self.dc.workingserialreadall()
                if r:
                    self.sres('[duringwriting] ')
                    self.sres(str(r))

                    # if self.sresplotmode == 1:
                    #    self.sres('Plotting ON')
                    #    self.sresPLOT(str(r))

        if not bsuppressendcode:
            self.dc.writebytes(b'\r\x04')
            self.dc.receivestream(bseekokay=True, isplotting=self.sresplotmode)

    def sendcommand(self, cellcontents):
        bsuppressendcode = False  # can't yet see how to get this signal through

        if self.srescapturedoutputfile:
            self.srescapturedoutputfile.close()  # shouldn't normally get here
            self.sres("closing stuck open srescapturedoutputfile\n")
            self.srescapturedoutputfile = None

        # extract any %-commands we have here at the start (or ending?),
        # tolerating pure comment lines and white space before the first %
        # (if there's no %-command in there, then no lines at the front get
        # dropped due to being comments)
        while True:
            mpercentline = re.match(
                "(?:(?:\s*|(?:\s*#.*\n))*)(%.*)\n?(?:[ \r]*\n)?",
                cellcontents)
            if not mpercentline:
                break

            # discards the %command and a single blank line
            # (if there is one) from the cell contents

            cellcontents = self.interpretpercentline(
                mpercentline.group(1),
                cellcontents[mpercentline.end():])

            if (isinstance(cellcontents, dict)
                    and cellcontents.get("source") == "set_next_input"):
                return cellcontents  # set_next_input_payload:
            if cellcontents is None:
                return None

        if not self.dc.serialexists():
            self.sres("No serial connected\n", 31)
            self.sres("  %serialconnect to connect\n")
            self.sres("  %esptool to flash the device\n")
            self.sres("  %lsmagic to list commands")
            return None

        # run the cell contents as normal
        if cellcontents:
            self.runnormalcell(cellcontents, bsuppressendcode)
        return None

    async def do_execute(
            self,
            code,
            silent,
            store_history=True,
            user_expressions=None,
            allow_stdin=False,
            *,
            cell_id=None,
    ):
        """Handle code execution."""
        self.use_micropython = True  # Assume true

        # extract any %-commands we have here at the start (or ending?),
        # tolerating pure comment lines and white space before the first %
        # (if there's no %-command in there, then no lines at the front get
        # dropped due to being comments)
        for _ in range(10):
            mpercentline = re.match(
                "(?:(?:\s*|(?:\s*#.*\n))*)(%.*)\n?(?:[ \r]*\n)?",
                code)
            if not mpercentline:
                break

            # discards the %command and a single blank line
            # (if there is one) from the cell contents
            percentline = mpercentline.group(1)
            try:
                percentstringargs = shlex.split(percentline)
            except ValueError as e:
                self.sres("\n\n***Bad percentcommand [%s]\n" % str(e), 31)
                self.sres(percentline)
                return None

            percentcommand = percentstringargs[0]

            if percentcommand == ap_bypass.prog:
                self.use_micropython = False
                code = code[mpercentline.end():]
                break

        if self.use_micropython:
            self.silent = silent
            if not code.strip():
                return {'status': 'ok', 'execution_count': self.execution_count,
                        'payload': [], 'user_expressions': {}}

            interrupted = False

            # clear buffer out before executing any commands (except the readbytes one)
            if self.dc.serialexists() and not re.match(
                    "\s*%readbytes|\s*%disconnect|\s*%serialconnect|\s*websocketconnect",
                    code):
                priorbuffer = None
                try:
                    priorbuffer = self.dc.workingserialreadall()
                except KeyboardInterrupt:
                    interrupted = True
                except OSError as e:
                    priorbuffer = []
                    self.sres(
                        "\n\n***Connection broken [%s]\n" % str(e.strerror), 31)
                    self.sres("You may need to reconnect")
                    self.dc.disconnect(raw=True, verbose=True)

                # except websocket.WebSocketConnectionClosedException as e:
                #     priorbuffer = []
                #     self.sres("\n\n***Websocket connection broken [%s]\n" % str(
                #         e.strerror), 31)
                #     self.sres("You may need to reconnect")
                #     self.dc.disconnect(raw=True, verbose=True)

                if priorbuffer:
                    if type(priorbuffer) == bytes:
                        try:
                            priorbuffer = priorbuffer.decode()
                        except UnicodeDecodeError:
                            priorbuffer = str(priorbuffer)

                    for pbline in priorbuffer.splitlines():
                        if deviceconnector.wifimessageignore.match(pbline):
                            continue  # filter out boring wifi status messages
                        if pbline:
                            self.sres('[leftinbuffer] ')
                            self.sres(str([pbline]))
                            self.sres('\n')

            set_next_input_payload = None
            try:
                if not interrupted:
                    set_next_input_payload = self.sendcommand(code)
            except KeyboardInterrupt:
                interrupted = True
            except OSError as e:
                self.sres("\n\n***OSError [%s]\n\n" % str(e.strerror))
            # except pexpect.EOF:
            #    self.sres(self.asyncmodule.before + 'Restarting Bash')
            #    self.startasyncmodule()

            if self.sresplotmode == 1:  # matplotlib-eqsue plotting (after finishing cell)
                self.sendPLOT()
            self.sresPLOTkiller()

            if self.srescapturedoutputfile:
                if self.srescapturemode == 2:
                    self.send_response(self.iopub_socket, 'clear_output',
                                       {"wait": True})
                if self.srescapturemode == 2 or self.srescapturemode == 3:
                    output = "{} lines captured.".format(
                        self.srescapturedlinecount)  # finish off by updating with the correct number captured
                    stream_content = {'name': "stdout", 'text': output}
                    self.send_response(self.iopub_socket, 'stream',
                                       stream_content)

                self.srescapturedoutputfile.close()
                self.srescapturedoutputfile = None
                self.srescapturemode = 0

            if interrupted:
                self.sresSYS("\n\n*** Sending Ctrl-C\n\n")
                if self.dc.serialexists():
                    self.dc.writebytes(b'\r\x03')
                    interrupted = True
                    try:
                        self.dc.receivestream(bseekokay=False,
                                              b5secondtimeout=True)
                    except KeyboardInterrupt:
                        self.sres(
                            "\n\nKeyboard interrupt while waiting response on Ctrl-C\n\n")
                    except OSError as e:
                        self.sres(
                            "\n\n***OSError while issuing a Ctrl-C [%s]\n\n" % str(
                                e.strerror))
                return {'status': 'abort',
                        'execution_count': self.execution_count}

            # everything already gone out with send_response(), but could detect errors (text between the two \x04s

            payload = [
                set_next_input_payload] if set_next_input_payload else []  # {"source": "set_next_input", "text": "some cell content", "replace": False}
            return {'status': 'ok', 'execution_count': self.execution_count,
                    'payload': payload, 'user_expressions': {}}

        else:
            shell = self.shell  # we'll need this a lot here

            self._forward_input(allow_stdin)

            reply_content: t.Dict[str, t.Any] = {}
            if hasattr(shell, "run_cell_async") and hasattr(shell,
                                                            "should_run_async"):
                run_cell = shell.run_cell_async
                should_run_async = shell.should_run_async
                with_cell_id = _accepts_cell_id(run_cell)
            else:
                should_run_async = lambda cell: False  # noqa

                # older IPython,
                # use blocking run_cell and wrap it in coroutine

                async def run_cell(*args, **kwargs):
                    return shell.run_cell(*args, **kwargs)

                with_cell_id = _accepts_cell_id(shell.run_cell)
            try:
                # default case: runner is asyncio and asyncio is already running
                # TODO: this should check every case for "are we inside the runner",
                # not just asyncio
                preprocessing_exc_tuple = None
                try:
                    transformed_cell = self.shell.transform_cell(code)
                except Exception:
                    transformed_cell = code
                    preprocessing_exc_tuple = sys.exc_info()

                if (
                        _asyncio_runner
                        and shell.loop_runner is _asyncio_runner
                        and asyncio.get_event_loop().is_running()
                        and should_run_async(
                    code,
                    transformed_cell=transformed_cell,
                    preprocessing_exc_tuple=preprocessing_exc_tuple,
                )
                ):
                    if with_cell_id:
                        coro = run_cell(
                            code,
                            store_history=store_history,
                            silent=silent,
                            transformed_cell=transformed_cell,
                            preprocessing_exc_tuple=preprocessing_exc_tuple,
                            cell_id=cell_id,
                        )
                    else:
                        coro = run_cell(
                            code,
                            store_history=store_history,
                            silent=silent,
                            transformed_cell=transformed_cell,
                            preprocessing_exc_tuple=preprocessing_exc_tuple,
                        )

                    coro_future = asyncio.ensure_future(coro)

                    with self._cancel_on_sigint(coro_future):
                        res = None
                        try:
                            res = await coro_future
                        finally:
                            shell.events.trigger("post_execute")
                            if not silent:
                                shell.events.trigger("post_run_cell", res)
                else:
                    # runner isn't already running,
                    # make synchronous call,
                    # letting shell dispatch to loop runners
                    if with_cell_id:
                        res = shell.run_cell(
                            code,
                            store_history=store_history,
                            silent=silent,
                            cell_id=cell_id,
                        )
                    else:
                        res = shell.run_cell(code, store_history=store_history,
                                             silent=silent)
            finally:
                self._restore_input()

            err = res.error_before_exec if res.error_before_exec is not None else res.error_in_exec

            if res.success:
                reply_content["status"] = "ok"
            else:
                reply_content["status"] = "error"

                reply_content.update(
                    {
                        "traceback": shell._last_traceback or [],
                        "ename": str(type(err).__name__),
                        "evalue": str(err),
                    }
                )

                # FIXME: deprecated piece for ipyparallel (remove in 5.0):
                e_info = dict(engine_uuid=self.ident, engine_id=self.int_id,
                              method="execute")
                reply_content["engine_info"] = e_info

            # Return the execution counter so clients can display prompts
            reply_content["execution_count"] = shell.execution_count - 1

            if "traceback" in reply_content:
                self.log.info(
                    "Exception in execute request:\n%s",
                    "\n".join(reply_content["traceback"]),
                )

            # At this point, we can tell whether the main code execution succeeded
            # or not.  If it did, we proceed to evaluate user_expressions
            if reply_content["status"] == "ok":
                reply_content["user_expressions"] = shell.user_expressions(
                    user_expressions or {})
            else:
                # If there was an error, don't even try to compute expressions
                reply_content["user_expressions"] = {}

            # Payloads should be retrieved regardless of outcome, so we can both
            # recover partial output (that could have been generated early in a
            # block, before an error) and always clear the payload system.
            reply_content["payload"] = shell.payload_manager.read_payload()
            # Be aggressive about clearing the payload because we don't want
            # it to sit in memory until the next execute_request comes in.
            shell.payload_manager.clear_payload()

            return reply_content

    def do_complete(self, code, cursor_pos):
        """Handle code completion."""
        if _use_experimental_60_completion and self.use_experimental_completions:
            return self._experimental_do_complete(code, cursor_pos)

        # FIXME: IPython completers currently assume single line,
        # but completion messages give multi-line context
        # For now, extract line from cell, based on cursor_pos:
        if cursor_pos is None:
            cursor_pos = len(code)
        line, offset = line_at_cursor(code, cursor_pos)
        line_cursor = cursor_pos - offset
        txt, matches = self.shell.complete("", line, line_cursor)
        return {
            "matches": matches,
            "cursor_end": cursor_pos,
            "cursor_start": cursor_pos - len(txt),
            "metadata": {},
            "status": "ok",
        }

    async def do_debug_request(self, msg):
        """Handle a debug request."""
        if _is_debugpy_available:
            return await self.debugger.process_request(msg)

    def _experimental_do_complete(self, code, cursor_pos):
        """
        Experimental completions from IPython, using Jedi.
        """
        if cursor_pos is None:
            cursor_pos = len(code)
        with _provisionalcompleter():
            raw_completions = self.shell.Completer.completions(code, cursor_pos)
            completions = list(_rectify_completions(code, raw_completions))

            comps = []
            for comp in completions:
                comps.append(
                    dict(
                        start=comp.start,
                        end=comp.end,
                        text=comp.text,
                        type=comp.type,
                        signature=comp.signature,
                    )
                )

        if completions:
            s = completions[0].start
            e = completions[0].end
            matches = [c.text for c in completions]
        else:
            s = cursor_pos
            e = cursor_pos
            matches = []

        return {
            "matches": matches,
            "cursor_end": e,
            "cursor_start": s,
            "metadata": {_EXPERIMENTAL_KEY_NAME: comps},
            "status": "ok",
        }

    def do_inspect(self, code, cursor_pos, detail_level=0, omit_sections=()):
        """Handle code inspection."""
        name = token_at_cursor(code, cursor_pos)

        reply_content: t.Dict[str, t.Any] = {"status": "ok"}
        reply_content["data"] = {}
        reply_content["metadata"] = {}
        try:
            if release.version_info >= (8,):
                # `omit_sections` keyword will be available in IPython 8, see
                # https://github.com/ipython/ipython/pull/13343
                bundle = self.shell.object_inspect_mime(
                    name,
                    detail_level=detail_level,
                    omit_sections=omit_sections,
                )
            else:
                bundle = self.shell.object_inspect_mime(name,
                                                        detail_level=detail_level)
            reply_content["data"].update(bundle)
            if not self.shell.enable_html_pager:
                reply_content["data"].pop("text/html")
            reply_content["found"] = True
        except KeyError:
            reply_content["found"] = False

        return reply_content

    def do_history(
            self,
            hist_access_type,
            output,
            raw,
            session=0,
            start=0,
            stop=None,
            n=None,
            pattern=None,
            unique=False,
    ):
        """Handle code history."""
        if hist_access_type == "tail":
            hist = self.shell.history_manager.get_tail(
                n, raw=raw, output=output, include_latest=True
            )

        elif hist_access_type == "range":
            hist = self.shell.history_manager.get_range(
                session, start, stop, raw=raw, output=output
            )

        elif hist_access_type == "search":
            hist = self.shell.history_manager.search(
                pattern, raw=raw, output=output, n=n, unique=unique
            )
        else:
            hist = []

        return {
            "status": "ok",
            "history": list(hist),
        }

    def do_shutdown(self, restart):
        """Handle kernel shutdown."""
        self.shell.exit_now = True
        return dict(status="ok", restart=restart)

    def do_is_complete(self, code):
        """Handle an is_complete request."""
        transformer_manager = getattr(self.shell, "input_transformer_manager",
                                      None)
        if transformer_manager is None:
            # input_splitter attribute is deprecated
            transformer_manager = self.shell.input_splitter
        status, indent_spaces = transformer_manager.check_complete(code)
        r = {"status": status}
        if status == "incomplete":
            r["indent"] = " " * indent_spaces
        return r

    def do_apply(self, content, bufs, msg_id, reply_metadata):
        """Handle an apply request."""
        try:
            from ipyparallel.serialize import serialize_object, \
                unpack_apply_message
        except ImportError:
            from .serialize import serialize_object, unpack_apply_message

        shell = self.shell
        try:
            working = shell.user_ns

            prefix = "_" + str(msg_id).replace("-", "") + "_"
            f, args, kwargs = unpack_apply_message(bufs, working, copy=False)

            fname = getattr(f, "__name__", "f")

            fname = prefix + "f"
            argname = prefix + "args"
            kwargname = prefix + "kwargs"
            resultname = prefix + "result"

            ns = {fname: f, argname: args, kwargname: kwargs, resultname: None}
            # print ns
            working.update(ns)
            code = f"{resultname} = {fname}(*{argname},**{kwargname})"
            try:
                exec(code, shell.user_global_ns, shell.user_ns)  # noqa
                result = working.get(resultname)
            finally:
                for key in ns:
                    working.pop(key)

            result_buf = serialize_object(
                result,
                buffer_threshold=self.session.buffer_threshold,
                item_threshold=self.session.item_threshold,
            )

        except BaseException as e:
            # invoke IPython traceback formatting
            shell.showtraceback()
            reply_content = {
                "traceback": shell._last_traceback or [],
                "ename": str(type(e).__name__),
                "evalue": str(e),
            }
            # FIXME: deprecated piece for ipyparallel (remove in 5.0):
            e_info = dict(engine_uuid=self.ident, engine_id=self.int_id,
                          method="apply")
            reply_content["engine_info"] = e_info

            self.send_response(
                self.iopub_socket,
                "error",
                reply_content,
                ident=self._topic("error"),
            )
            self.log.info("Exception in apply request:\n%s",
                          "\n".join(reply_content["traceback"]))
            result_buf = []
            reply_content["status"] = "error"
        else:
            reply_content = {"status": "ok"}

        return reply_content, result_buf

    def do_clear(self):
        """Clear the kernel."""
        self.shell.reset(False)
        return dict(status="ok")


# This exists only for backwards compatibility - use IPythonKernel instead


class Kernel(IPythonKernel):
    """DEPRECATED.  An alias for the IPython kernel class."""

    def __init__(self, *args, **kwargs):  # pragma: no cover
        """DEPRECATED."""
        import warnings

        warnings.warn(
            "Kernel is a deprecated alias of alpaca_kernel.ipkernel.IPythonKernel",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
