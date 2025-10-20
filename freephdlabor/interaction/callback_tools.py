# socket_user_input.py
import socket
import threading
from queue import Queue, Empty
from typing import Callable, Iterable

from smolagents.memory import TaskStep, PlanningStep
from .user_inststep import UserInstructionStep


def setup_user_input_socket(host: str = "127.0.0.1", port: int = 5001) -> Queue[str]:
    """
    Start a tiny TCP server that accepts multiple client connections (sequential or concurrent).
    Anything typed by any connected client is pushed (line-by-line) into the returned Queue[str].
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)  # allow a small backlog
    print(f"[Info] Awaiting user input on {host}:{port}")
    print(f"[Info] Connect from another terminal:  nc {host} {port}")

    input_queue: Queue[str] = Queue()
    stop_flag = threading.Event()

    def socket_listener(sock: socket.socket, client_addr, queue: Queue[str]):
        buf = ""
        try:
            print(f"[Info] Connected: {client_addr}")
            while True:
                data = sock.recv(1024)
                if not data:
                    break
                buf += data.decode()
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    queue.put(line.rstrip("\r"))  # preserve empty lines
        except Exception as e:
            print(f"[Warn] Client {client_addr} listener error: {e}")
        finally:
            try:
                sock.close()
            finally:
                print(f"[Info] Input connection closed: {client_addr}")

    def accept_loop():
        try:
            while not stop_flag.is_set():
                try:
                    client_socket, client_addr = server_socket.accept()
                except OSError:
                    # server_socket likely closed
                    break
                threading.Thread(
                    target=socket_listener,
                    args=(client_socket, client_addr, input_queue),
                    daemon=True,
                ).start()
        finally:
            try:
                server_socket.close()
            except OSError:
                pass
            print("[Info] Server socket closed")

    threading.Thread(target=accept_loop, daemon=True).start()
    return input_queue


def make_user_input_step_callback(
    input_queue: Queue[str],
    interrupt_signals: Iterable[str] | None = None,
) -> Callable:
    """
    Build a step_callback that pauses on interrupt, asks for user input,
    clarifies if it's a 'modification' or a 'new' task, and appends the
    corresponding step to the agent's memory before resuming.
    """
    interrupt_set = set((interrupt_signals or ["interrupt", "stop", "pause"]))
    paused = False

    def _try_get_nowait(q: Queue[str]):
        try:
            return q.get_nowait()
        except Empty:
            return None

    def _read_until_double_enter(q: Queue[str], banner: str) -> str:
        print(banner)
        print(">>> Type your instruction. Press Enter twice to finish.\n")
        lines: list[str] = []
        empty_streak = 0
        while empty_streak < 2:
            line = q.get()
            if line is None:
                line = ""
            if line.strip() == "":
                empty_streak += 1
            else:
                empty_streak = 0
            lines.append(line)
        # Strip the final two empties
        while lines and lines[-1].strip() == "":
            lines.pop()
            if lines and lines[-1].strip() == "":
                lines.pop()
                break
        return "\n".join(lines).strip()

    def callback(memory_step, agent):
        nonlocal paused

        # If not paused, check for an interrupt signal.
        if not paused:
            cmd = _try_get_nowait(input_queue)
            if cmd and cmd.strip().lower() in (s.lower() for s in interrupt_set):
                paused = True
                print("\n🛑 Interrupt received — pausing for user instruction...")
        
        if not paused:
            return  # No interrupt, carry on.

        # If paused, get user input and clarify intent.
        print("\n" + "=" * 60)
        print("📝 WAITING FOR USER INSTRUCTION")
        print("=" * 60)

        instruction = _read_until_double_enter(input_queue, banner="--- PROVIDE INSTRUCTION (leave empty to cancel) ---")

        if instruction:
            # Ask for clarification
            print("\nIs this a 'modification' to the current task or a 'new' task? (m/n)")
            choice = ""
            while choice not in ["m", "n"]:
                choice = input_queue.get().strip().lower()
                if choice not in ["m", "n"]:
                    print("Invalid choice. Please enter 'm' for modification or 'n' for new task.")

            try:
                if choice == "m":
                    # Create and append a UserInstructionStep for modification
                    new_step = UserInstructionStep(user_instruction=instruction)
                    agent.memory.steps.append(new_step)
                    print("\n✅ User modification appended to memory.")
                else: # choice == "n"
                    # Create and append a standard TaskStep for a new task
                    new_step = TaskStep(task=instruction)
                    agent.memory.steps.append(new_step)
                    print("\n✅ New user task appended to memory.")
            except Exception as e:
                print(f"\n❌ Failed to add user instruction to memory: {e}")
        else:
            print("\nℹ️ No instruction provided. Resuming without change.")

        paused = False
        print("✅ Resuming...\n")
        return

    return callback
