package com.example;

import java.util.HashMap;
import java.util.Map;

public class PrefabInletChatSpace {

    private Map<String, String> chatSpaceConfig;

    public PrefabInletChatSpace() {
        chatSpaceConfig = new HashMap<>();
        initializeChatSpace();
    }

    private void initializeChatSpace() {
        // Configuration for Pieces OS
        chatSpaceConfig.put("PiecesOS", "Configuration for Pieces OS");

        // Configuration for OpenEuler OS
        chatSpaceConfig.put("OpenEulerOS", "Configuration for OpenEuler OS");
    }

    public String getChatSpaceConfig(String osName) {
        return chatSpaceConfig.getOrDefault(osName, "Configuration not found for the specified OS");
    }

    public static void main(String[] args) {
        PrefabInletChatSpace chatSpace = new PrefabInletChatSpace();
        System.out.println(chatSpace.getChatSpaceConfig("PiecesOS"));
        System.out.println(chatSpace.getChatSpaceConfig("OpenEulerOS"));
    }
}
