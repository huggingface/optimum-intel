<script>
  export let command = '';
  let showMsg = false;

  function copy() {
    navigator.clipboard.writeText(command);
    showMsg = true;
    clearTimeout(window._copyTimeout);
    window._copyTimeout = setTimeout(() => (showMsg = false), 1000);
  }
</script>

<button on:click={copy}>
  ✅
</button>

{#if showMsg}
  <span style="margin-left: 10px;">Copied!</span>
{/if}