# Keeps Windows from going to sleep while the live site is up (started and stopped by go-live.ps1 / stop-live.ps1).
# It only asks Windows "do not sleep while this program runs" — no setting is changed, and the request ends with this process.
Add-Type -Namespace Win -Name Power -MemberDefinition '[System.Runtime.InteropServices.DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS_SYSTEM_REQUIRED = [uint32]2147483649      # ES_CONTINUOUS (0x80000000) | ES_SYSTEM_REQUIRED (0x00000001)
while ($true) {
  [void][Win.Power]::SetThreadExecutionState($ES_CONTINUOUS_SYSTEM_REQUIRED)
  Start-Sleep -Seconds 30
}
