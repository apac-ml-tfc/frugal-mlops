# Python Built-Ins:
from datetime import datetime
from dateutil.relativedelta import relativedelta
import signal
import time
from typing import Any, Callable, Optional
import warnings


def wait(
    fn_poll_result: Callable[[], Any],
    fn_is_finished: Callable[[Any], bool],
    fn_stringify_result: Optional[Callable[[Any], str]]=None,
    spinner_secs: float=0.4,
    poll_secs: float=30
) -> None:
    """Polling-wait with loading spinner and elapsed time indicator.

    Displays a spinner, the current stringified status, and the elapsed time since the wait was started or
    the stringified status last changed (as a dateutil.relativedelta down to 1sec precision). E.g:

    / Status: InProgress - AnalyzingData [Since: relativedelta(minutes=+10, seconds=+33)]

    A new line is generated with reset "Since", every time the status string changes. No limit on line
    length, but generated status string must not contain newlines.

    Parameters
    ----------
    fn_poll_result :
        Zero-argument callable that returns some kind of job status descriptor
    fn_is_finished :
        Returns True if job completed, False if ongoing, raises error if failed
    fn_stringify_result : Optional
        Status object stringifier for the console output [defaults to str(status)]
    spinner_secs : Optional
        Time to sleep between check cycles
    poll_secs : Optional
        Minimum elapsed time since last poll after which next check cycle will call fn_poll_result
    """
    SPINNER_STATES = ("/", "-", "\\", "|")
    status = fn_poll_result()
    status_t0 = datetime.now().replace(microsecond=0)
    status_str = fn_stringify_result(status) if fn_stringify_result else str(status)
    t = 0
    i = 0
    maxlen = 0
    print(f"Initial status: {status_str}")
    while not fn_is_finished(status):
        if t >= poll_secs:
            newstatus = fn_poll_result()
            newstatus_str = fn_stringify_result(newstatus) if fn_stringify_result else str(newstatus)
            t = t - poll_secs
            if status_str == newstatus_str:
                print("\r", end="")
            else:
                print("\n", end="")
                status_t0 = datetime.now().replace(microsecond=0)
            status = newstatus
            status_str = newstatus_str
        else:
            print("\r", end="")
        i = (i + 1) % len(SPINNER_STATES)
        msg = "{} Status: {} [Since: {}]".format(
            SPINNER_STATES[i],
            status_str,
            relativedelta(datetime.now().replace(microsecond=0), status_t0)
        )
        maxlen = max(maxlen, len(msg))
        msg = msg.ljust(maxlen)
        print(msg, end="")
        time.sleep(spinner_secs)
        t += spinner_secs
    print("")
    return status


def notebook_safe_tqdm_loop(tqdm_iterator, fn):
    """Construct a tqdm progress bar-decorated for loop safe for Jupyter(Lab) notebooks

    Erroring out of a tqdm-decorated for loop (e.g. due to cell interrupt or an exception) without closing
    the tqdm context will mess up future calls and even accumulate (every open context causes an additional
    newline to be output between all pbar updates).

    This function constructs a safe context to execute fn against each object in the iterator.

    Parameters
    ----------
    tqdm_iterator : Iterable
        A tqdm-decorated iterator, e.g. tqdm.tqdm(range(10))
    fn : Callable[Any, Any]
        A function to call sequentially with each object in the iterator
    """
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    def signal_handler(signal, frame):
        try:
            tqdm_iterator.close()
        except:
            warnings.warn("notebook_safe_tqdm_loop couldn't close tqdm_iterator context")
        return original_sigint_handler(signal, frame)

    try:
        signal.signal(signal.SIGINT, signal_handler)
        for obj in tqdm_iterator:
            result = fn(obj)
        return result
    except Exception as e:
        # For a non-signal exception:
        iterator.close()
        raise e
    finally:
        # Restore original SIGINT handler:
        signal.signal(signal.SIGINT, original_sigint_handler)
