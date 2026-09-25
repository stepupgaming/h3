use anyhow::Result;
use std::process::Child;

/// Owns exactly one supervised Windows process tree. Closing the handle kills
/// descendants that are still attached; explicit cancellation uses
/// `terminate` after the cooperative grace period.
#[cfg(windows)]
pub(crate) struct OwnedProcessTree {
    handle: windows_sys::Win32::Foundation::HANDLE,
}

#[cfg(windows)]
unsafe impl Send for OwnedProcessTree {}

#[cfg(windows)]
impl OwnedProcessTree {
    pub(crate) fn create() -> Result<Self> {
        use anyhow::Context;
        use std::mem::{size_of, zeroed};
        use windows_sys::Win32::System::JobObjects::{
            CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
            SetInformationJobObject,
        };

        let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if handle.is_null() {
            return Err(std::io::Error::last_os_error()).context("create Windows Job Object");
        }
        let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { zeroed() };
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let configured = unsafe {
            SetInformationJobObject(
                handle,
                JobObjectExtendedLimitInformation,
                std::ptr::addr_of!(limits).cast(),
                size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };
        if configured == 0 {
            let error = std::io::Error::last_os_error();
            unsafe {
                windows_sys::Win32::Foundation::CloseHandle(handle);
            }
            return Err(error).context("configure Windows Job Object kill-on-close");
        }
        Ok(Self { handle })
    }

    pub(crate) fn assign(&self, child: &Child) -> Result<()> {
        use anyhow::Context;
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::System::JobObjects::AssignProcessToJobObject;

        let assigned = unsafe {
            AssignProcessToJobObject(self.handle, child.as_raw_handle() as *mut std::ffi::c_void)
        };
        if assigned == 0 {
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("assign pid {} to Windows Job Object", child.id()));
        }
        Ok(())
    }

    pub(crate) fn terminate(&self, exit_code: u32) -> Result<()> {
        use anyhow::Context;
        use windows_sys::Win32::System::JobObjects::TerminateJobObject;

        let terminated = unsafe { TerminateJobObject(self.handle, exit_code) };
        if terminated == 0 {
            return Err(std::io::Error::last_os_error())
                .context("terminate owned Windows Job Object");
        }
        Ok(())
    }
}

#[cfg(windows)]
impl Drop for OwnedProcessTree {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                windows_sys::Win32::Foundation::CloseHandle(self.handle);
            }
            self.handle = std::ptr::null_mut();
        }
    }
}

#[cfg(not(windows))]
pub(crate) struct OwnedProcessTree;

#[cfg(not(windows))]
impl OwnedProcessTree {
    pub(crate) fn create() -> Result<Self> {
        Ok(Self)
    }

    pub(crate) fn assign(&self, _child: &Child) -> Result<()> {
        Ok(())
    }

    pub(crate) fn terminate(&self, _exit_code: u32) -> Result<()> {
        Ok(())
    }
}
