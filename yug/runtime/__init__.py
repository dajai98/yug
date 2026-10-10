"""YUG Core Runtime: the first composition layer connecting Brain,
Memory, and Tools.

Enforced dependency direction (this package depends on the others;
none of them depend back on this one or on each other):

    yug.runtime -> yug.brain.interface.BrainInterface (abstraction only)
    yug.runtime -> yug.memory.working_memory.WorkingMemory
    yug.runtime -> yug.tools.executor.ToolExecutor

This is deliberately NOT an agent: tool execution only happens when a
caller explicitly names a tool and its parameters (``respond(...,
tool_name=..., tool_parameters=...)``). The brain never selects,
requests, or triggers a tool call itself -- there is no code path from
a brain's ``GenerationResult.tool_calls`` field to actual execution in
this milestone. Adding that is explicitly future Agent Runtime scope.
"""
