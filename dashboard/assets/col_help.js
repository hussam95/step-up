(function () {
  const HELP_SELECTOR = "details.col-help";
  let closeTimer = null;

  function clearTimer() {
    if (closeTimer) {
      window.clearTimeout(closeTimer);
      closeTimer = null;
    }
  }

  function closeAll(except) {
    document.querySelectorAll(HELP_SELECTOR + "[open]").forEach((el) => {
      if (el !== except) {
        el.open = false;
      }
    });
  }

  function armAutoClose(target) {
    clearTimer();
    closeTimer = window.setTimeout(() => {
      target.open = false;
      closeTimer = null;
    }, 3000);
  }

  function toggleHelp(details) {
    const shouldOpen = !details.open;
    closeAll(details);
    clearTimer();
    details.open = shouldOpen;
    if (shouldOpen) {
      armAutoClose(details);
    }
  }

  document.addEventListener(
    "click",
    (event) => {
      const summary = event.target.closest(`${HELP_SELECTOR} > summary`);
      if (summary) {
        event.preventDefault();
        event.stopPropagation();
        toggleHelp(summary.parentElement);
        return;
      }

      if (!event.target.closest(HELP_SELECTOR)) {
        closeAll(null);
        clearTimer();
      }
    },
    true
  );

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeAll(null);
      clearTimer();
    }
  });
})();
