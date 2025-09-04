import { create } from 'zustand'

export const useWorkflowStore = create((set, get) => ({
  currentStep: 'data',
  steps: ['data', 'features', 'model', 'train', 'backtest', 'evaluate', 'deploy'],
  
  setCurrentStep: (step) => {
    console.log('Setting current step to:', step)
    set({ currentStep: step })
  },
  
  nextStep: () => set((state) => {
    const currentIndex = state.steps.indexOf(state.currentStep)
    const nextIndex = Math.min(currentIndex + 1, state.steps.length - 1)
    const nextStep = state.steps[nextIndex]
    console.log('Moving to next step:', nextStep)
    return { currentStep: nextStep }
  }),
  
  previousStep: () => set((state) => {
    const currentIndex = state.steps.indexOf(state.currentStep)
    const prevIndex = Math.max(currentIndex - 1, 0)
    const prevStep = state.steps[prevIndex]
    console.log('Moving to previous step:', prevStep)
    return { currentStep: prevStep }
  }),
  
  canGoNext: () => {
    const state = get()
    const currentIndex = state.steps.indexOf(state.currentStep)
    return currentIndex < state.steps.length - 1
  },
  
  canGoPrevious: () => {
    const state = get()
    const currentIndex = state.steps.indexOf(state.currentStep)
    return currentIndex > 0
  },
}))
