# AI Plan - Feb 11



we want to trigger a motion matching search when the inputs change. When the desired velocity is different than the previous desired velocity, by more than 1 m/s, then we schedule a search for 0.02 seconds in the future. This lets the player finish the push on the thumbstick. We can trigger such an "input decided search" only once every "inputDecidedSearchPeriod" seconds, which defaults to 0.2 seconds. The normal search timer is restarted when an input decided search is triggered.